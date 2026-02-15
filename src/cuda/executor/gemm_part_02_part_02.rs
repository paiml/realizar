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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
}
