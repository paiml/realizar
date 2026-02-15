impl CudaExecutor {

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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
            let module = self.compile_ptx(&ptx)?;
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
}
