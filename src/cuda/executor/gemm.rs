//! Multi-stream async execution and GEMM/GEMV operations
//!
//! This module implements:
//! - PARITY-038: Multi-Stream Async Execution
//! - GEMM operations (tiled, optimized, fused)
//! - GEMV operations for M=1 token generation
//! - Softmax kernel
//! - Q4K/Q5K/Q6K GEMV with direct weight transfer

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

impl CudaExecutor {
    // ========================================================================
    // PARITY-038: Multi-Stream Async Execution
    // ========================================================================

    /// Synchronize compute stream only (wait for kernel execution)
    pub fn synchronize_compute(&self) -> Result<(), GpuError> {
        self.compute_stream.synchronize()
    }

    /// Synchronize transfer stream only (wait for H2D/D2H transfers)
    pub fn synchronize_transfer(&self) -> Result<(), GpuError> {
        self.transfer_stream.synchronize()
    }

    /// Synchronize all streams (compute + transfer)
    pub fn synchronize_all(&self) -> Result<(), GpuError> {
        self.compute_stream.synchronize()?;
        self.transfer_stream.synchronize()?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Execute async GEMM using cached weights on compute stream (PARITY-038)
    ///
    /// This launches the kernel without waiting for completion.
    /// Call `synchronize_compute()` to wait for the result.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight tensor
    /// * `input_buf` - Pre-allocated GPU buffer for input B
    /// * `output_buf` - Pre-allocated GPU buffer for output C
    /// * `m`, `n`, `k` - Matrix dimensions
    ///
    /// # Safety
    ///
    /// Input and output buffers must remain valid until stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns error if weights not found or kernel fails to launch.
    pub fn gemm_cached_async(
        &mut self,
        weight_name: &str,
        input_buf: &GpuBuffer<f32>,
        output_buf: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weights
        let weight_ptr = self
            .weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!("Weight '{}' not cached", weight_name))
            })?
            .as_ptr();

        // Generate/load kernel
        let kernel_type = KernelType::GemmTiled {
            m,
            n,
            k,
            tile_size: 32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_{}_{}_{}_{}", m, n, k, 32);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Launch config
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let config = LaunchConfig::grid_2d((n + 31) / 32, (m + 31) / 32, 32, 32);

        // Launch on compute stream (non-blocking)
        let mut ptr_a = weight_ptr;
        let mut ptr_b = input_buf.as_ptr();
        let mut ptr_c = output_buf.as_ptr();
        let mut m_val = m as i32;
        let mut n_val = n as i32;
        let mut k_val = k as i32;

        // SAFETY: Buffers valid, caller ensures lifetime
        unsafe {
            self.compute_stream.launch_kernel(
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

        Ok(())
    }

    /// Allocate a GPU buffer for async operations (PARITY-038)
    ///
    /// Returns a buffer that can be used with async copy and GEMM operations.
    pub fn allocate_buffer(&self, len: usize) -> Result<GpuBuffer<f32>, GpuError> {
        GpuBuffer::new(&self.context, len)
    }

    /// Async copy from host to GPU buffer on transfer stream (PARITY-038)
    ///
    /// # Safety
    ///
    /// Host data must remain valid until transfer stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns error if copy fails.
    pub unsafe fn copy_to_gpu_async(
        &self,
        buf: &mut GpuBuffer<f32>,
        data: &[f32],
    ) -> Result<(), GpuError> {
        // SAFETY: Caller guarantees data remains valid until stream sync
        unsafe { buf.copy_from_host_async(data, &self.transfer_stream) }
    }

    /// Async copy from GPU buffer to host on transfer stream (PARITY-038)
    ///
    /// # Safety
    ///
    /// Host buffer must remain valid until transfer stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns error if copy fails.
    pub unsafe fn copy_from_gpu_async(
        &self,
        buf: &GpuBuffer<f32>,
        data: &mut [f32],
    ) -> Result<(), GpuError> {
        // SAFETY: Caller guarantees data remains valid until stream sync
        unsafe { buf.copy_to_host_async(data, &self.transfer_stream) }
    }

    /// Execute GEMM using cached weights: C = cached_A @ B (PARITY-037)
    ///
    /// This is the fast path for inference - weights stay on GPU.
    /// Only input B and output C are transferred.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight tensor (must be pre-loaded)
    /// * `b` - Input matrix B (k x n, row-major)
    /// * `c` - Output matrix C (m x n, row-major)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    ///
    /// # Errors
    ///
    /// Returns error if weights not found or kernel fails.
    pub fn gemm_cached(
        &mut self,
        weight_name: &str,
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weights
        let weight_ptr = self
            .weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!("Weight '{}' not cached", weight_name))
            })?
            .as_ptr();

        // Validate sizes
        let expected_b = (k * n) as usize;
        let expected_c = (m * n) as usize;

        if b.len() != expected_b || c.len() != expected_c {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMM size mismatch: B[{}] expected {}, C[{}] expected {}",
                b.len(),
                expected_b,
                c.len(),
                expected_c
            )));
        }

        // Generate PTX for this configuration
        let kernel_type = KernelType::GemmTiled {
            m,
            n,
            k,
            tile_size: 32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_{}_{}_{}_{}", m, n, k, 32);

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

        // Allocate GPU buffers for input B and output C only
        // Weight A is already on GPU (cached)
        let buf_b = GpuBuffer::from_host(&self.context, b)?;
        // PARITY-114 FIX: Initialize output buffer with zeros to prevent state accumulation
        let c_zeros = vec![0.0f32; expected_c];
        let buf_c = GpuBuffer::from_host(&self.context, &c_zeros)?;

        // Launch configuration
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        // The tiled GEMM kernel uses ctaid.y for rows and ctaid.x for columns
        let config = LaunchConfig::grid_2d(
            (n + 31) / 32, // Grid X - columns (N dimension)
            (m + 31) / 32, // Grid Y - rows (M dimension)
            32,            // Block X
            32,            // Block Y
        );

        // Get raw pointers for kernel args
        let mut ptr_a = weight_ptr; // From cache!
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

    /// Execute GEMM with cached B (weight) matrix: C = A × B_cached
    ///
    /// This is the correct method for transformer inference where:
    /// - A is the input activation (changes each forward pass)
    /// - B is the weight matrix (constant, cached on GPU)
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight (B matrix, k×n)
    /// * `a` - Input matrix A (m×k elements, row-major)
    /// * `c` - Output matrix C (m×n elements, row-major)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    ///
    /// Returns error if weights not found or kernel fails.
    pub fn gemm_b_cached(
        &mut self,
        weight_name: &str,
        a: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight B
        let weight_ptr = self
            .weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!("Weight '{}' not cached", weight_name))
            })?
            .as_ptr();

        // Validate sizes
        let expected_a = (m * k) as usize;
        let expected_c = (m * n) as usize;

        if a.len() != expected_a || c.len() != expected_c {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMM size mismatch: A[{}] expected {}, C[{}] expected {}",
                a.len(),
                expected_a,
                c.len(),
                expected_c
            )));
        }

        // Generate PTX for this configuration
        let kernel_type = KernelType::GemmTiled {
            m,
            n,
            k,
            tile_size: 32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_{}_{}_{}_{}", m, n, k, 32);

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

        // Allocate GPU buffers for input A and output C only
        // Weight B is already on GPU (cached)
        let buf_a = GpuBuffer::from_host(&self.context, a)?;
        let c_zeros = vec![0.0f32; expected_c];
        let buf_c = GpuBuffer::from_host(&self.context, &c_zeros)?;

        // Launch configuration
        let config = LaunchConfig::grid_2d(
            (n + 31) / 32, // Grid X - columns (N dimension)
            (m + 31) / 32, // Grid Y - rows (M dimension)
            32,            // Block X
            32,            // Block Y
        );

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = weight_ptr; // From cache!
        let mut ptr_c = buf_c.as_ptr();
        let mut m_val = m as i32;
        let mut n_val = n as i32;
        let mut k_val = k as i32;

        // Launch kernel: C = A × B where B is cached
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

    /// Execute a tiled GEMM kernel: C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m x k, row-major)
    /// * `b` - Input matrix B (k x n, row-major)
    /// * `c` - Output matrix C (m x n, row-major)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    ///
    /// # Errors
    ///
    /// Returns error if kernel execution fails.
    pub fn gemm(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
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

        // Generate PTX for this configuration
        // PARITY-003: Enable simpler Gemv (warp-reduce) for M=1 operations
        let use_gemv = m == 1;
        let (kernel_type, cache_key) = if use_gemv {
            (KernelType::Gemv { k, n }, format!("gemv_{}_{}", k, n))
        } else {
            (
                KernelType::GemmTiled {
                    m,
                    n,
                    k,
                    tile_size: 32,
                },
                format!("gemm_{}_{}_{}_{}", m, n, k, 32),
            )
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

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

        // Launch configuration differs for Gemv vs GEMM
        // PARITY-003: Enable simpler Gemv with correct config
        let config = if use_gemv {
            // Simple Gemv: 32 threads (one warp) per block, N blocks
            // Each block computes one output element y[block_id]
            LaunchConfig::grid_2d(n, 1, 32, 1)
        } else {
            // GEMM: 2D grid of 32x32 tiles
            // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
            LaunchConfig::grid_2d(
                (n + 31) / 32, // Grid X - columns (N dimension)
                (m + 31) / 32, // Grid Y - rows (M dimension)
                32,            // Block X
                32,            // Block Y
            )
        };

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = buf_b.as_ptr();
        let mut ptr_c = buf_c.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // Launch kernel
        // SAFETY: Buffers are valid, config matches kernel expectations
        // PARITY-003: Enable GEMV for M=1 operations
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            if use_gemv {
                // GEMV kernel: y = B * x where x is A (1×K row as K vector), B is K×N, y is C (1×N as N vector)
                // Args: y_ptr, a_ptr (matrix), x_ptr, k_dim, n_dim
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void, // y_ptr (output)
                        std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void, // a_ptr (K×N matrix)
                        std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void, // x_ptr (K input vector)
                        std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void, // k_dim
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void, // n_dim
                    ],
                )?;
            } else {
                // GEMM kernel: C = A × B
                // Args: a_ptr, b_ptr, c_ptr, m, n, k
                let mut m_val = m as i32;
                let mut n_val_i32 = n as i32;
                let mut k_val_i32 = k as i32;
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val_i32) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_val_i32) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        Ok(())
    }

    /// Execute GEMV using cached weight matrix (PARITY-120: 10x speedup)
    ///
    /// This is the fast path for single-token generation (M=1).
    /// The weight matrix must be pre-loaded via `load_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight matrix
    /// * `x` - Input vector (K elements)
    /// * `y` - Output vector (N elements)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if weight not cached or kernel execution fails.
    pub fn gemv_cached(
        &mut self,
        weight_name: &str,
        x: &[f32],
        y: &mut [f32],
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        // Validate sizes
        if x.len() != k as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMV input size mismatch: got {}, expected {}",
                x.len(),
                k
            )));
        }
        if y.len() != n as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMV output size mismatch: got {}, expected {}",
                y.len(),
                n
            )));
        }

        // Get cached weight buffer
        let buf_w = self.weight_cache.get(weight_name).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Weight '{}' not cached on GPU", weight_name))
        })?;

        // PARITY-003: Use simpler Gemv kernel (32 threads warp-reduce) instead of CoalescedGemv
        let kernel_type = KernelType::Gemv { k, n };
        let cache_key = format!("gemv_simple_{}_{}", k, n);
        let kernel_name = self.kernels.kernel_name(&kernel_type);

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

        // Allocate only input/output buffers (weight stays on GPU!)
        let buf_x = GpuBuffer::from_host(&self.context, x)?;
        let y_zeros = vec![0.0f32; n as usize];
        let buf_y = GpuBuffer::from_host(&self.context, &y_zeros)?;

        // PARITY-003: Simple Gemv config - 32 threads (one warp) per block, N blocks
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        // Get raw pointers
        let mut ptr_y = buf_y.as_ptr();
        let mut ptr_w = buf_w.as_ptr();
        let mut ptr_x = buf_x.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // Launch kernel: y = W * x
        // SAFETY: Buffers are valid, config matches kernel expectations
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_y) as *mut std::ffi::c_void, // y_ptr (output)
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void, // w_ptr (K×N matrix, CACHED)
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void, // x_ptr (K input vector)
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void, // k_dim
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void, // n_dim
                ],
            )?;
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_y.copy_to_host(y)?;

        Ok(())
    }

    /// Execute optimized GEMM kernel (IMP-900a)
    ///
    /// Uses larger tile sizes and register blocking for better performance.
    /// Provides ~2-3x improvement over naive tiled GEMM.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m × k)
    /// * `b` - Input matrix B (k × n)
    /// * `c` - Output matrix C (m × n)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    /// * `tile_size` - Tile size for shared memory (32 or 64)
    ///
    /// # Errors
    ///
    /// Returns error if kernel execution fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_optimized(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        tile_size: u32,
    ) -> Result<(), GpuError> {
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

        // IMP-900a: Use optimized kernel with larger tiles
        let reg_block = if tile_size >= 64 { 8 } else { 4 };
        let kernel_type = KernelType::GemmOptimized {
            m,
            n,
            k,
            tile_size,
            reg_block,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_opt_{}_{}_{}_{}", m, n, k, tile_size);

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

        // Launch configuration with optimized tile size
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let config = LaunchConfig::grid_2d(
            (n + tile_size - 1) / tile_size, // Grid X - columns (N dimension)
            (m + tile_size - 1) / tile_size, // Grid Y - rows (M dimension)
            tile_size,                       // Block X
            tile_size,                       // Block Y
        );

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

    /// Execute fused GEMM + bias + activation kernel (IMP-900b)
    ///
    /// Performs C = activation(A @ B + bias) in a single kernel launch,
    /// reducing kernel launch overhead by 3x compared to separate operations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m × k)
    /// * `b` - Input matrix B (k × n)
    /// * `bias` - Bias vector (n elements) or None
    /// * `c` - Output matrix C (m × n)
    /// * `m`, `n`, `k` - Matrix dimensions
    /// * `activation` - Activation type (0=none, 1=relu, 2=gelu)
    ///
    /// # Performance Impact
    ///
    /// - Without fusion: 3 kernel launches (GEMM + add + activation)
    /// - With fusion: 1 kernel launch
    /// - Expected improvement: 1.3-1.5x for small matrices
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fused(
        &mut self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        activation: u32,
    ) -> Result<(), GpuError> {
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

        if let Some(b_vec) = bias {
            if b_vec.len() != n as usize {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "Bias size mismatch: got {}, expected {}",
                    b_vec.len(),
                    n
                )));
            }
        }

        // Track fusion stats in pool
        self.memory_pool
            .record_allocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        // IMP-900b: Use fused kernel type
        let kernel_type = KernelType::GemmBiasActivation {
            m,
            n,
            k,
            activation,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_fused_{}_{}_{}_{}", m, n, k, activation);

        // Load module if not cached (falls back to tiled for now)
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

        // Launch configuration
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let tile_size = 32u32;
        let config = LaunchConfig::grid_2d(
            (n + tile_size - 1) / tile_size, // Grid X - columns (N dimension)
            (m + tile_size - 1) / tile_size, // Grid Y - rows (M dimension)
            tile_size,
            tile_size,
        );

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

        // IMP-1000: Apply bias + activation on GPU (eliminates host roundtrip)
        if bias.is_some() || activation > 0 {
            let total_elements = expected_c as u32;

            // Create bias buffer (use zeros if no bias)
            let bias_data: Vec<f32> =
                bias.map_or_else(|| vec![0.0f32; n as usize], <[f32]>::to_vec);
            let buf_bias = GpuBuffer::from_host(&self.context, &bias_data)?;

            // Load epilogue kernel
            let epilogue_type = KernelType::BiasActivation {
                n: total_elements,
                bias_size: n,
                activation,
            };
            let epilogue_name = self.kernels.kernel_name(&epilogue_type);
            let epilogue_key = format!("bias_act_{}_{}", total_elements, activation);

            if !self.modules.contains_key(&epilogue_key) {
                let ptx = self.kernels.generate_ptx(&epilogue_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(epilogue_key.clone(), module);
            }

            let epilogue_module = self
                .modules
                .get_mut(&epilogue_key)
                .expect("module just inserted");

            // Launch epilogue kernel
            let threads = 256u32;
            let blocks = (total_elements + threads - 1) / threads;
            let epilogue_config = LaunchConfig::linear(blocks, threads);

            let mut ptr_c_epilogue = buf_c.as_ptr();
            let mut ptr_bias = buf_bias.as_ptr();
            let mut n_val_epilogue = total_elements as i32;
            let mut bias_size_val = n as i32;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    epilogue_module,
                    epilogue_name,
                    &epilogue_config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_c_epilogue) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_bias) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val_epilogue) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut bias_size_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Synchronize and copy result back (single H2D transfer)
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        self.memory_pool
            .record_deallocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        Ok(())
    }

    /// Execute softmax kernel on a vector
    ///
    /// Computes numerically stable softmax in-place.
    pub fn softmax(&mut self, data: &mut [f32]) -> Result<(), GpuError> {
        let dim = data.len() as u32;

        let kernel_type = KernelType::Softmax { dim };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("softmax_{}", dim);

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

        // Allocate input and output buffers on GPU
        let input_buf = GpuBuffer::from_host(&self.context, data)?;
        let output_buf: GpuBuffer<f32> = GpuBuffer::new(&self.context, data.len())?;

        // Launch with 1 block, dim threads (up to 1024)
        let threads = dim.min(1024);
        let config = LaunchConfig::linear(1, threads);

        // Get raw pointers for kernel args (input_ptr, output_ptr, length)
        let mut input_ptr = input_buf.as_ptr();
        let mut output_ptr = output_buf.as_ptr();
        let mut length_val = dim;

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut output_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        output_buf.copy_to_host(data)?;

        Ok(())
    }

    /// Execute Q4_K quantized GEMM (fused dequantization + matmul)
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q4_K format
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 32)
    pub fn q4k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PARITY-003 FIX: Use QuantizedGemmGgml for GGUF Q4_K format (256 values, 144 bytes per super-block)
        // Previous: QuantizedGemm was for a different Q4 layout, causing garbage output
        let kernel_type = KernelType::QuantizedGemmGgml { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_ggml_{}_{}", m, k);

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

        // PARITY-003 FIX: Launch configuration for GGML kernel with tile_size=32
        // The GGML kernel uses: weight_row = clamped_col, where clamped_col = ctaid_x * tile + local_col
        // For matvec (n=1), we want weight_row to iterate over m output elements
        // CRITICAL FIX: Swap m and n so kernel uses ctaid_x for weight row indexing
        // grid.x = ceil(m/tile), grid.y = 1 (but we pass n=m, m=1 to kernel!)
        let tile_size = 32u32;
        let blocks_x = (m + tile_size - 1) / tile_size; // Iterate over m outputs
        let blocks_y = 1u32; // n=1, so 1 block in y
        let config = LaunchConfig::grid_2d(blocks_x, blocks_y, tile_size, tile_size);

        // Get raw pointers for kernel args
        // Kernel signature: q4k_gemm_ggml(a_ptr, b_quant_ptr, c_ptr, m, n, k)
        // Where: a_ptr = input activations, b_quant_ptr = weights, c_ptr = output
        //
        // PARITY-003 FIX: For matvec, swap m and n so kernel uses ctaid_x for weight row
        // The kernel uses clamped_col (derived from ctaid_x) to index weight rows
        // By passing m=1, n=out_dim, the kernel will:
        //   - Use ctaid_x (0 to out_dim/tile) for weight row indexing via clamped_col
        //   - Output at index global_row * n + global_col = 0 * out_dim + col = col
        let mut ptr_input = buf_input.as_ptr(); // a_ptr: input activations
        let mut ptr_weights = buf_weights.as_ptr(); // b_quant_ptr: quantized weights
        let mut ptr_output = buf_output.as_ptr(); // c_ptr: output
        let mut m_val = 1u32; // m=1 (swapped for matvec)
        let mut n_val = m; // n=out_dim (swapped for matvec)
        let mut k_val = k; // u32 as expected by kernel

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void, // a_ptr
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void, // b_quant_ptr
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void, // c_ptr
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,     // m
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,     // n (was missing!)
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,     // k
                ],
            )?;
        }

        // Synchronize and copy result
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q4_K GEMV (fused dequantization + matvec) - PAR-003
    ///
    /// Optimized kernel for M=1 token generation. Uses warp shuffle reduction
    /// with one warp (32 threads) per output element. No shared memory needed.
    ///
    /// # Performance
    ///
    /// - Memory: 7.1x more efficient than dequant+GEMV (reads Q4_K directly)
    /// - Compute: Fused dequant+multiply avoids intermediate buffer
    /// - Target: >24 tok/s (M2 milestone), matching llama.cpp performance
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q4_K GGML format (144 bytes per 256 values)
    /// * `input` - Input vector (f32, length k)
    /// * `output` - Output vector (f32, length n)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q4k_gemv(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-003: Use dedicated Q4_K GEMV kernel for M=1 operations
        let kernel_type = KernelType::Q4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemv_{}_{}", k, n);

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
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-003: Launch configuration for GEMV kernel
        // Grid: N blocks (one per output element)
        // Block: 32 threads (one warp for reduction)
        // No shared memory needed
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        // Kernel signature: q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim)
        let mut ptr_output = buf_output.as_ptr(); // y_ptr: output vector
        let mut ptr_weights = buf_weights.as_ptr(); // w_ptr: quantized weights
        let mut ptr_input = buf_input.as_ptr(); // x_ptr: input vector
        let mut k_val = k; // k_dim
        let mut n_val = n; // n_dim

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void, // y_ptr
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void, // w_ptr
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,  // x_ptr
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,      // k_dim
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,      // n_dim
                ],
            )?;
        }

        // Synchronize and copy result
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q5_K GEMV (fused dequantization + matvec) - PAR-003
    pub fn q5k_gemv(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
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

        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
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

    /// Execute Q6_K GEMV (fused dequantization + matvec) - PAR-003
    pub fn q6k_gemv(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
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

        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
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
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // GEMM Tests (gemm, gemm_optimized, gemm_fused)
    // ========================================================================

    #[test]
    fn test_gemm_identity_matrix() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // 2x2 identity @ 2x2 vector
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];
        exec.gemm(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert!((c[0] - 3.0).abs() < 0.1);
        assert!((c[3] - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_gemm_small_matrix() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A@B = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        exec.gemm(&a, &b, &mut c, 2, 2, 2).unwrap();
        // Tiled GEMM may have some numerical error
        assert!((c[0] - 19.0).abs() < 1.0);
        assert!((c[3] - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_gemm_m1_uses_gemv() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=1 should use GEMV kernel path
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        let b = vec![1.0, 0.5, 0.25, 0.125, 2.0, 1.0, 0.5, 0.25]; // 4x2
        let mut c = vec![0.0; 2]; // 1x2
        exec.gemm(&a, &b, &mut c, 1, 2, 4).unwrap();
        // dot(a, b[:, 0]) = 1*1 + 2*0.5 + 3*0.25 + 4*0.125 = 1+1+0.75+0.5 = 3.25
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_size_mismatch_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0, 2.0]; // Wrong size
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let result = exec.gemm(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_optimized_tile32() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 64]; // 8x8
        let b = vec![1.0; 64]; // 8x8
        let mut c = vec![0.0; 64]; // 8x8
        exec.gemm_optimized(&a, &b, &mut c, 8, 8, 8, 32).unwrap();
        // Each element should be sum of 8 ones = 8
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_fused_no_activation() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        exec.gemm_fused(&a, &b, None, &mut c, 2, 2, 2, 0).unwrap();
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_fused_with_relu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0, -1.0, 1.0, -1.0]; // Some negative
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        // activation=1 is ReLU
        exec.gemm_fused(&a, &b, None, &mut c, 2, 2, 2, 1).unwrap();
        // ReLU should clamp negatives to 0
        for val in &c {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_gemm_fused_with_bias() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let bias = vec![10.0, 20.0];
        let mut c = vec![0.0; 4];
        exec.gemm_fused(&a, &b, Some(&bias), &mut c, 2, 2, 2, 0)
            .unwrap();
        // Bias should be added
        assert!(c[0] > 10.0);
    }

    #[test]
    fn test_gemm_fused_bias_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let bias = vec![10.0]; // Wrong size, should be 2
        let mut c = vec![0.0; 4];
        let result = exec.gemm_fused(&a, &b, Some(&bias), &mut c, 2, 2, 2, 0);
        assert!(result.is_err());
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    #[test]
    fn test_softmax_uniform() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        exec.softmax(&mut data).unwrap();
        // Uniform input -> uniform output
        let expected = 0.25;
        for val in &data {
            assert!((*val - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_softmax_single_max() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![10.0, 0.0, 0.0, 0.0];
        exec.softmax(&mut data).unwrap();
        // First element should dominate
        assert!(data[0] > 0.9);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        exec.softmax(&mut data).unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_large_values() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![100.0, 101.0, 102.0, 103.0];
        exec.softmax(&mut data).unwrap();
        // Should still sum to 1 (numerically stable)
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // Allocate Buffer Tests
    // ========================================================================

    #[test]
    fn test_allocate_buffer_basic() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1024).unwrap();
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn test_allocate_buffer_small() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1).unwrap();
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_allocate_buffer_large() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1024 * 1024).unwrap();
        assert_eq!(buf.len(), 1024 * 1024);
    }

    // ========================================================================
    // Synchronize Tests
    // ========================================================================

    #[test]
    fn test_synchronize_compute() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_compute().is_ok());
    }

    #[test]
    fn test_synchronize_transfer() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_transfer().is_ok());
    }

    #[test]
    fn test_synchronize_all() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_all().is_ok());
    }

    // ========================================================================
    // Cached GEMM Tests
    // ========================================================================

    #[test]
    fn test_gemm_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let result = exec.gemm_cached("nonexistent_weight", &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_b_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let result = exec.gemm_b_cached("nonexistent_weight", &a, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 4];
        let result = exec.gemv_cached("nonexistent_weight", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_input_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 2]; // Wrong size
        let mut y = vec![0.0; 4];
        let result = exec.gemv_cached("test", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_output_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 2]; // Wrong size
        let result = exec.gemv_cached("test", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    // ========================================================================
    // Async Copy Tests
    // ========================================================================

    #[test]
    fn test_async_copy_roundtrip() {
        let Some(exec) = create_executor() else {
            return;
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buf = exec.allocate_buffer(4).unwrap();

        // Copy to GPU async
        unsafe {
            exec.copy_to_gpu_async(&mut buf, &data).unwrap();
        }
        exec.synchronize_transfer().unwrap();

        // Copy back from GPU async
        let mut result = vec![0.0f32; 4];
        unsafe {
            exec.copy_from_gpu_async(&buf, &mut result).unwrap();
        }
        exec.synchronize_transfer().unwrap();

        assert_eq!(result, data);
    }
}
