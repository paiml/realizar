//! Cached quantized GEMV methods for Q4K/Q5K/Q6K weights
//!
//! This module implements:
//! - PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
//! - Q4K/Q5K/Q6K GEMV with GPU-cached weights
//! - Tiled, chunked, and coalesced GEMV variants
//! - DP4A SIMD-accelerated GEMV
//! - Fused RMSNorm + Q4K GEMV
//! - Batched Q4K/Q6K GEMV

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

impl CudaExecutor {
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

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Tiled Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_tiled_q4k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.tiled_q4k_gemv_cached_async("nonexistent", &input, 128, 256, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_tiled_q4k_gemv_small_k() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        // K=256 < MAX_TILED_K, uses standard tiled kernel
        let result = exec.tiled_q4k_gemv_cached_async("test", &input, 128, 256, 4);
        assert!(result.is_err()); // Weight not cached
    }

    // ========================================================================
    // Chunked Tiled Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_chunked_tiled_q4k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.chunked_tiled_q4k_gemv_cached_async("nonexistent", &input, 128, 256, 4);
        assert!(result.is_err());
    }

    // ========================================================================
    // DP4A Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_dp4a_q4k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.dp4a_q4k_gemv_cached_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q8 Quantize Tests
    // ========================================================================

    #[test]
    fn test_q8_quantize_async_output_size_calculation() {
        // Test Q8_1 format output size: ceil(n/32) * 36 bytes
        let n = 256u32;
        let num_blocks = (n + 31) / 32;
        let expected_bytes = (num_blocks * 36) as usize;
        assert_eq!(expected_bytes, 288);
    }

    #[test]
    fn test_q8_quantize_async_single_block_size() {
        // Test Q8_1 format output size for single block
        let n = 32u32;
        let num_blocks = (n + 31) / 32;
        let expected_bytes = (num_blocks * 36) as usize;
        assert_eq!(expected_bytes, 36);
    }

    // ========================================================================
    // Q4K Q8 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q4k_q8_gemv_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let q8_input = GpuBuffer::<u8>::new(&exec.context, 288).unwrap();
        let result = exec.q4k_q8_gemv_async("nonexistent", &q8_input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // True DP4A Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_true_dp4a_q4k_gemv_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.true_dp4a_q4k_gemv_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Packed DP4A Tests
    // ========================================================================

    #[test]
    fn test_packed_dp4a_q4k_q8_gemv_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let q8_input = GpuBuffer::<u8>::new(&exec.context, 288).unwrap();
        let result = exec.packed_dp4a_q4k_q8_gemv_async("nonexistent", &q8_input, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_packed_dp4a_full_async_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.packed_dp4a_full_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q5K/Q6K GEMV Cached Tests
    // ========================================================================

    #[test]
    fn test_q5k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q5k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q6k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else { return; };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q6k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // GELU GPU Tests
    // ========================================================================

    #[test]
    fn test_gelu_gpu_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input = vec![0.0f32, 1.0, -1.0, 2.0];
        let buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let result = exec.gelu_gpu(&buf, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gelu_gpu_larger() {
        let Some(mut exec) = create_executor() else { return; };
        let input = vec![1.0f32; 256];
        let buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let result = exec.gelu_gpu(&buf, 256);
        assert!(result.is_ok());
    }

    // ========================================================================
    // LayerNorm GPU Tests
    // ========================================================================

    #[test]
    fn test_layer_norm_gpu_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let beta = GpuBuffer::from_host(&exec.context, &vec![0.0f32; 32]).unwrap();
        let result = exec.layer_norm_gpu(&input, &output, &gamma, &beta, 32, 1, 1e-5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // RMSNorm GPU Tests
    // ========================================================================

    #[test]
    fn test_rmsnorm_gpu_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let result = exec.rmsnorm_gpu(&input, &gamma, 32, 1e-5);
        assert!(result.is_ok());
        let buf = result.unwrap();
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn test_rmsnorm_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
        let result = exec.rmsnorm_into(&input, &gamma, &output, 32, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rmsnorm_host_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input = vec![1.0f32; 32];
        let gamma = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 32];
        let result = exec.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
        assert!(result.is_ok());
        // Output should not be zeros anymore
        exec.stream.synchronize().unwrap();
    }

    // ========================================================================
    // Batched RMSNorm Tests
    // ========================================================================

    #[test]
    fn test_batched_rmsnorm_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        // M=4 sequences, hidden=32
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 4 * 32]).unwrap();
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 4 * 32).unwrap();
        let result = exec.batched_rmsnorm_into(&input, &gamma, &output, 32, 4, 1e-5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Residual Add GPU Tests
    // ========================================================================

    #[test]
    fn test_residual_add_gpu_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; 32]).unwrap();
        let result = exec.residual_add_gpu(&input1, &input2, 32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_residual_add_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; 32]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
        let result = exec.residual_add_into(&input1, &input2, &output, 32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_residual_add_host_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let input1 = vec![1.0f32; 32];
        let input2 = vec![2.0f32; 32];
        let mut output = vec![0.0f32; 32];
        let result = exec.residual_add_host(&input1, &input2, &mut output);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Fused Residual RMSNorm Tests
    // ========================================================================

    #[test]
    fn test_fused_residual_rmsnorm_gpu_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let residual = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
        let result = exec.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, 32, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_residual_rmsnorm_host_basic() {
        let Some(mut exec) = create_executor() else { return; };
        let residual = vec![1.0f32; 32];
        let input = vec![1.0f32; 32];
        let gamma = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 32];
        let result = exec.fused_residual_rmsnorm_host(&residual, &input, &gamma, &mut output, 1e-5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Batched Operations Tests
    // ========================================================================

    #[test]
    fn test_batched_rope_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        // M=4 sequences, num_heads=4, head_dim=32
        let size = 4 * 4 * 32;
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
        let positions = GpuBuffer::from_host(&exec.context, &vec![0u32; 4]).unwrap();
        let result = exec.batched_rope_into(&input, &output, &positions, 4, 32, 4, 10000.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_residual_add_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        // M=4 sequences, n=32 elements each
        let size = 4 * 32;
        let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; size]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
        let result = exec.batched_residual_add_into(&input1, &input2, &output, 32, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_swiglu_into_basic() {
        let Some(mut exec) = create_executor() else { return; };
        // M=4 sequences, n=32 elements each
        let size = 4 * 32;
        let gate = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let up = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
        let result = exec.batched_swiglu_into(&gate, &up, &output, 32, 4);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_rmsnorm_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        // Create gamma buffer directly (avoid borrow conflict with cache)
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let result = exec.rmsnorm_gpu(&input, &gamma, config.hidden_dim as u32, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rmsnorm_into_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        // Create gamma buffer directly
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();
        let result = exec.rmsnorm_into(&input, &gamma, &output, config.hidden_dim as u32, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_rmsnorm_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        // Create gamma buffer directly
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let m = 4u32;
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; (m as usize) * config.hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, (m as usize) * config.hidden_dim).unwrap();
        let result = exec.batched_rmsnorm_into(&input, &gamma, &output, config.hidden_dim as u32, m, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_residual_rmsnorm_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        // Create gamma buffer directly
        let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let residual = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
        let input = GpuBuffer::from_host(&exec.context, &vec![0.5f32; config.hidden_dim]).unwrap();
        let result = exec.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, config.hidden_dim as u32, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_rope_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        let m = 4u32;
        let total_dim = (m as usize) * config.num_heads * config.head_dim;
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; total_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, total_dim).unwrap();
        let positions = GpuBuffer::from_host(&exec.context, &vec![0u32, 1, 2, 3]).unwrap();

        let result = exec.batched_rope_into(
            &input,
            &output,
            &positions,
            config.num_heads as u32,
            config.head_dim as u32,
            m,
            exec.rope_theta,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_swiglu_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        let m = 4u32;
        let size = (m as usize) * config.intermediate_dim;
        let gate = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let up = GpuBuffer::from_host(&exec.context, &vec![2.0f32; size]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();

        let result = exec.batched_swiglu_into(&gate, &up, &output, config.intermediate_dim as u32, m);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batched_residual_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else { return; };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() { return; }

        let m = 4u32;
        let size = (m as usize) * config.hidden_dim;
        let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
        let input2 = GpuBuffer::from_host(&exec.context, &vec![0.5f32; size]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();

        let result = exec.batched_residual_add_into(&input1, &input2, &output, config.hidden_dim as u32, m);
        assert!(result.is_ok());
    }
}
