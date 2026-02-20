impl CudaExecutor {

    /// PAR-082-V2: Multi-warp Vectorized Q4K GEMV (4 warps + u32 coalesced loads)
    ///
    /// Combines VectorizedQ4K's coalesced u32 loads with multi-warp parallelism.
    /// 128 threads (4 warps) per block, matching llama.cpp's mul_mat_vec_q.
    pub fn mwv_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "mwv_q4k_gemv_into")?;
        // PAR-082-V3: Configurable warp count via MWV_WARPS env var (default: 2)
        let num_warps = crate::cuda::kernels::mwv_warp_count();
        let kernel_type = KernelType::MwvQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_q4k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // num_warps * 32 threads per output element, one block per output
        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, input) are allocated by
        // CudaExecutor and valid for the kernel's grid dimensions. k_val and n_val
        // are stack-local scalars passed by pointer. The stream synchronizes before
        // any host-side read of the output buffer.
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

    /// PAR-082-V4: Execute MWV DP4A Q4_K GEMV with Q8_1-quantized activations
    ///
    /// Two-step pipeline:
    /// 1. Quantize f32 activations → Q8_1 format (Q8QuantizeKernel)
    /// 2. DP4A dot product: Q4K weights × Q8_1 activations → f32 output
    ///
    /// 3.3x instruction reduction vs MWV, coalesced Q8 loads vs scattered f32.
    /// Enable with `DP4A_Q4K=1` env var.
    pub fn mwv_dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "mwv_dp4a_q4k_gemv_into")?;

        // PAR-PERF-DP4A: Use pre-allocated Q8 buffer from workspace (zero allocation)
        // Five-Whys root cause: Old code did GpuBuffer::new per call = 280 cudaMallocs/token
        // Fix: Borrow workspace.q8_activation_buf, quantize in-place, then launch GEMV
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("PAR-PERF-DP4A: workspace.q8_activation_buf not initialized. Call init_workspace() first.")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        // Create non-owning view into the pre-allocated Q8 buffer
        // SAFETY: q8_activation_buf is pre-allocated in init_workspace and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize activations to Q8_1 (into pre-allocated buffer)
        self.q8_quantize_into(input, &q8_buf, k)?;

        // Step 2: Launch DP4A GEMV kernel
        let num_warps = crate::cuda::kernels::mwv_warp_count();
        let kernel_type = KernelType::MwvDp4aQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_dp4a_q4k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8 = q8_buf.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, q8_buf) are allocated by
        // CudaExecutor and valid for the kernel's grid dimensions. k_val and n_val
        // are stack-local scalars passed by pointer. q8_buf is a view into
        // workspace.q8_activation_buf and is kept alive via std::mem::forget below.
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Don't drop — q8_buf is a view into workspace.q8_activation_buf
        std::mem::forget(q8_buf);

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
}
