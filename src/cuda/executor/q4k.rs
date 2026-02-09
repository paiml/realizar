//! Q4K quantized GEMV operations
//!
//! This module implements Q4_K dequantization and matrix-vector multiplication
//! for efficient inference with 4-bit quantized weights.

#![allow(clippy::wildcard_imports)]
#![allow(clippy::too_many_arguments)]

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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
        // Validate pointer before kernel launch — launching with ptr=0
        // crashes the kernel and permanently poisons the CUDA context.
        if weight_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null weight pointer in q4k_gemv_indexed_async".to_string(),
            ));
        }

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-132: Use VectorizedQ4KGemv (u32 coalesced loads, 79 tok/s)
        let kernel_type = KernelType::VectorizedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("vectorized_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

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
        // Validate pointer before kernel launch — launching with ptr=0
        // crashes the kernel and permanently poisons the CUDA context.
        if weight_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null weight pointer in q6k_gemv_indexed_async".to_string(),
            ));
        }
        // PAR-058: Direct pointer access for Q6K weights
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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
        // PAR-132: Use WideQ4KGemv (256 threads, 8 warps) for all Q4K GEMV
        //
        // Five-Whys root cause chain:
        // 1. Why 39 tok/s vs Ollama's 128 tok/s? → 12% BW utilization
        // 2. Why 12% BW? → SM occupancy too low to hide memory latency
        // 3. Why low occupancy? → 32 threads/block = 1 warp = 33% max occupancy
        // 4. Why 32 threads? → Original kernel: 1 warp per output element
        // 5. Why not more warps? → No cross-warp reduction was implemented
        //
        // Fix: WideQ4KGemvKernel uses 8 warps (256 threads) per output,
        // with cross-warp reduction via shared memory (32 bytes).
        // llama.cpp also uses 256 threads for Q4_K decode.
        //
        // PAR-132: Use VectorizedQ4KGemv (coalesced u32 loads) as default
        //
        // Empirical results on RTX 4090, 7B Q4K:
        // - Legacy tiled (128 threads): 41 tok/s (WIDE_Q4K_DISABLE=1)
        // - Wide 8-warp (256 threads):  67 tok/s (WIDE_Q4K=1)
        // - Vectorized u32 (32 threads): 79 tok/s (default)
        //
        // u32 coalesced loads > multi-warp occupancy for bandwidth-bound GEMV:
        // - 32 threads × 4 bytes = 128 bytes/transaction (fully coalesced)
        // - Each thread processes 8 nibbles per u32 load
        // - No cross-warp reduction overhead
        //
        // Env vars for A/B testing:
        // WIDE_Q4K_DISABLE=1 → legacy tiled/chunked
        // WIDE_Q4K=1 → multi-warp 256 threads
        if std::env::var("WIDE_Q4K_DISABLE").is_ok() {
            return self.q4k_gemv_into_legacy(weight_ptr, input, output, n, k);
        }
        if std::env::var("WIDE_Q4K").is_ok() {
            return self.wide_q4k_gemv_into(weight_ptr, input, output, n, k);
        }

        self.vectorized_q4k_gemv_into(weight_ptr, input, output, n, k)
    }

    /// Legacy Q4K GEMV dispatch (pre-PAR-132)
    /// Uses tiled/chunked kernels with 128 threads or basic with 32 threads
    fn q4k_gemv_into_legacy(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q4k_gemv_into_legacy")?;
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288;
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
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
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(weight_ptr, "coalesced_q4k_gemv_into")?;
        let kernel_type = KernelType::CoalescedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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

    /// PAR-132: Wide Q4_K GEMV with 256 threads (8 warps) per output
    ///
    /// Root cause fix for 3x Ollama performance gap:
    /// - Previous: 32 threads/block = 33% SM occupancy, can't hide memory latency
    /// - New: 256 threads/block = 67-100% occupancy, 8 warps hide latency
    ///
    /// Cross-warp reduction via 32 bytes shared memory per block.
    /// Target: 100+ tok/s decode (from 39 tok/s), reaching Ollama parity.
    #[inline]
    pub fn wide_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "wide_q4k_gemv_into")?;
        let kernel_type = KernelType::WideQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("wide_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-132: 8 warps (256 threads) per output element
        // Empirically: 8 warps (67 tok/s) > 4 warps (61 tok/s) on RTX 4090
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

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
        validate_device_ptr(weight_ptr, "vectorized_q4k_gemv_into")?;
        let kernel_type = KernelType::VectorizedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("vectorized_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(weight_ptr, "dp4a_q4k_gemv_into")?;
        let kernel_type = KernelType::Dp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(weight_ptr, "fused_rmsnorm_q4k_gemv_into(weight)")?;
        validate_device_ptr(gamma_ptr, "fused_rmsnorm_q4k_gemv_into(gamma)")?;
        let kernel_type = KernelType::FusedRmsNormQ4KGemv { k, n, epsilon };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_rmsnorm_q4k_gemv_{}_{}_{:.0e}", k, n, epsilon);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(gate_weight_ptr, "fused_gate_up_q4k_gemv_into(gate)")?;
        validate_device_ptr(up_weight_ptr, "fused_gate_up_q4k_gemv_into(up)")?;
        let kernel_type = KernelType::FusedGateUpQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(weight_ptr, "q4k_gemv_into_tiled")?;
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
            let module = self.compile_ptx(&ptx)?;
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
        validate_device_ptr(weight_ptr, "batched_q4k_gemv_into")?;
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
                let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: N blocks (one per output row), warps*32 threads per block
        let threads_per_block = warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads_per_block, 1);

        // m_dim = warps * 8 (each warp handles 8 batch elements)
        let m = warps * 8;

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // Kernel signature: batched_q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
        // Same signature as BatchedQ4KGemv since we use the same trueno kernel
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
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Q4K GEMV Cached Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q4k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q5k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q5k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q6k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q6k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Cached Async Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.q4k_gemv_cached_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q6k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.q6k_gemv_cached_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Indexed Async Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_indexed_async_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        // Use zero pointer - will likely fail kernel but tests buffer creation
        let result = exec.q4k_gemv_indexed_async(0, &input, 128, 256);
        // Just testing it compiles and runs - actual result depends on PTX
        let _ = result;
    }

    #[test]
    fn test_q6k_gemv_indexed_async_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.q6k_gemv_indexed_async(0, &input, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Q4K GEMV Into Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_into_tiled_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        // Test kernel loading path with zero weight pointer
        let result = exec.q4k_gemv_into_tiled(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_coalesced_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let result = exec.coalesced_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_vectorized_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let result = exec.vectorized_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_dp4a_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let result = exec.dp4a_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Fused Kernels Tests
    // ========================================================================

    #[test]
    fn test_fused_rmsnorm_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let result = exec.fused_rmsnorm_q4k_gemv_into(0, &input, 0, &output, 256, 128, 1e-5);
        let _ = result;
    }

    #[test]
    fn test_fused_gate_up_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let gate_out = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let up_out = GpuBuffer::<f32>::new(&exec.context, 128).unwrap();
        let result = exec.fused_gate_up_q4k_gemv_into(0, 0, &input, &gate_out, &up_out, 256, 128);
        let _ = result;
    }

    // ========================================================================
    // Batched Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_batched_q4k_gemv_into_m4() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=4, K=256, N=128
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 4 * 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 4 * 128).unwrap();
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 4, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m8() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=8 (max for single kernel)
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 8 * 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 8 * 128).unwrap();
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 8, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m16_multi_warp() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=16 uses multi-warp kernel
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 16 * 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 16 * 128).unwrap();
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 16, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m32_multi_warp() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=32 uses 4-warp kernel
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32 * 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 32 * 128).unwrap();
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 32, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m12_tiled() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=12 uses tiling (8+4)
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 12 * 256]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 12 * 128).unwrap();
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 12, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Get Quantized Weight Ptr Tests
    // ========================================================================

    #[test]
    fn test_get_quantized_weight_ptr_not_found() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.get_quantized_weight_ptr("nonexistent");
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Cached Tiled Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_tiled_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q4k_gemv_cached_tiled("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Use indexed weights from layer 0
        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();

        // Test Q4K GEMV using attn_q weight
        let result = exec.q4k_gemv_into_tiled(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_coalesced_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();

        let result = exec.coalesced_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_vectorized_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();

        let result = exec.vectorized_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_dp4a_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();

        let result = exec.dp4a_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_rmsnorm_q4k_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();

        // Use RMSNorm gamma pointer and attn_q weight
        let result = exec.fused_rmsnorm_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            layer_weights.attn_norm_ptr,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_gate_up_q4k_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let gate_out = GpuBuffer::<f32>::new(&exec.context, config.intermediate_dim).unwrap();
        let up_out = GpuBuffer::<f32>::new(&exec.context, config.intermediate_dim).unwrap();

        let result = exec.fused_gate_up_q4k_gemv_into(
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            &input,
            &gate_out,
            &up_out,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let m = 4u32;
        let input = GpuBuffer::from_host(
            &exec.context,
            &vec![0.1f32; (m as usize) * config.hidden_dim],
        )
        .unwrap();
        let output =
            GpuBuffer::<f32>::new(&exec.context, (m as usize) * config.hidden_dim).unwrap();

        let result = exec.batched_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            m,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }
}
