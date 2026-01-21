//! Activation functions and element-wise GPU operations
//!
//! This module implements:
//! - PAR-023: Activation and Element-wise GPU Operations
//! - GELU, LayerNorm, RMSNorm kernels
//! - Residual add operations
//! - Batched RMSNorm, RoPE, SwiGLU
//! - Host convenience wrappers

use super::*;

impl CudaExecutor {
    // =========================================================================
    // PAR-023: Activation and Element-wise GPU Operations
    // =========================================================================

    /// PAR-023: SiLU activation on GPU buffer
    ///
    /// Computes: output[i] = input[i] * sigmoid(input[i])
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn silu_gpu(&mut self, input: &GpuBuffer<f32>, n: u32) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::Silu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("silu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // 256 threads per block for element-wise ops
        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-023: GELU activation on GPU buffer (async, returns new buffer)
    ///
    /// Computes approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// Unlike `gelu_gpu`, this returns a new buffer for async pipeline use.
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn gelu_async(
        &mut self,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::Gelu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gelu_async_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-023: Element-wise multiply on GPU buffers
    ///
    /// Computes: output[i] = input1[i] * input2[i]
    /// Used for gated activations in SwiGLU.
    ///
    /// # Returns
    ///
    /// GPU buffer with product (no sync - async)
    pub fn elementwise_mul_gpu(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::ElementwiseMul { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("elementwise_mul_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

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

        Ok(output)
    }

    /// PAR-023: Fused SwiGLU activation on GPU buffers
    ///
    /// Computes: output[i] = silu(gate[i]) * up[i]
    /// Combines SiLU activation and multiply in one memory pass.
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn fused_swiglu_gpu(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::FusedSwiglu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_swiglu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-044: Fused SwiGLU into existing buffer (zero-allocation, async)
    #[inline]
    pub fn fused_swiglu_into(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedSwiglu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_swiglu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-PERF-009: Fused Q/K/V projection on GPU
    ///
    /// Computes Q, K, V projections in a single kernel launch.
    /// Reduces kernel launch overhead from 3 launches to 1.
    ///
    /// # Arguments
    ///
    /// * `x` - Input hidden state [hidden_size]
    /// * `w_q` - Query weight matrix [hidden_size, hidden_size]
    /// * `w_k` - Key weight matrix [hidden_size, kv_dim]
    /// * `w_v` - Value weight matrix [hidden_size, kv_dim]
    /// * `out_q` - Output Q buffer [hidden_size]
    /// * `out_k` - Output K buffer [kv_dim]
    /// * `out_v` - Output V buffer [kv_dim]
    /// * `hidden_size` - Hidden dimension
    /// * `kv_dim` - KV dimension (for GQA, may differ from hidden_size)
    pub fn fused_qkv_into(
        &mut self,
        x: &GpuBuffer<f32>,
        w_q: &GpuBuffer<f32>,
        w_k: &GpuBuffer<f32>,
        w_v: &GpuBuffer<f32>,
        out_q: &GpuBuffer<f32>,
        out_k: &GpuBuffer<f32>,
        out_v: &GpuBuffer<f32>,
        hidden_size: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedQKV {
            hidden_size,
            kv_dim,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_qkv_{}_{}", hidden_size, kv_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: one block per output row (max of hidden_size for Q, kv_dim for K/V)
        // Block: 32 threads (one warp) per row
        let rows = hidden_size.max(kv_dim);
        let config = LaunchConfig::grid_2d(rows, 1, 32, 1);

        let mut ptr_x = x.as_ptr();
        let mut ptr_wq = w_q.as_ptr();
        let mut ptr_wk = w_k.as_ptr();
        let mut ptr_wv = w_v.as_ptr();
        let mut ptr_out_q = out_q.as_ptr();
        let mut ptr_out_k = out_k.as_ptr();
        let mut ptr_out_v = out_v.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wq) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wk) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wv) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_v) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-PERF-009: Fused Gate+Up FFN with SwiGLU on GPU
    ///
    /// Computes gate and up projections + SiLU activation in a single kernel.
    /// Reduces kernel launch overhead from 2 launches + activation to 1.
    ///
    /// # Arguments
    ///
    /// * `x` - Input hidden state [hidden_size]
    /// * `w_gate` - Gate weight matrix [hidden_size, intermediate_size]
    /// * `w_up` - Up weight matrix [hidden_size, intermediate_size]
    /// * `output` - Output buffer [intermediate_size], contains SiLU(gate) * up
    /// * `hidden_size` - Hidden dimension
    /// * `intermediate_size` - Intermediate FFN dimension
    pub fn fused_gate_up_into(
        &mut self,
        x: &GpuBuffer<f32>,
        w_gate: &GpuBuffer<f32>,
        w_up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        intermediate_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedGateUp {
            hidden_size,
            intermediate_size,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_{}_{}", hidden_size, intermediate_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: one block per output row (intermediate_size)
        // Block: 32 threads (one warp) per row
        let config = LaunchConfig::grid_2d(intermediate_size, 1, 32, 1);

        let mut ptr_x = x.as_ptr();
        let mut ptr_wg = w_gate.as_ptr();
        let mut ptr_wu = w_up.as_ptr();
        let mut ptr_out = output.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wg) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wu) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-060: Apply RoPE (Rotary Position Embedding) on GPU
    ///
    /// Applies rotary position embeddings to Q or K vectors.
    /// This is a critical optimization - eliminates CPU fallback that caused
    /// 28 GPU syncs + D2H/H2D copies per token.
    ///
    /// # Arguments
    ///
    /// * `input` - Input Q or K vector (FP32)
    /// * `output` - Output buffer (can alias input for in-place)
    /// * `position` - Current sequence position
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `theta` - RoPE base frequency
    pub fn rope_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Rope {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: 1 thread per rotation pair = num_heads * (head_dim / 2)
        let num_pairs = num_heads * (head_dim / 2);
        let threads = 256;
        let blocks = (num_pairs + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut pos_val = position;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-054: RoPE with indirect position (CUDA Graph Compatible)
    ///
    /// Same as `rope_into` but reads position from device memory instead of kernel parameter.
    /// This is required for CUDA graph capture since kernel parameters are baked at capture time.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (num_heads * head_dim elements)
    /// * `output` - Output tensor (same size as input)
    /// * `position_buf` - Device buffer containing u32 position (1 element)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `theta` - RoPE base frequency
    pub fn rope_indirect_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::RopeIndirect {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_indirect_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: 1 thread per rotation pair = num_heads * (head_dim / 2)
        let num_pairs = num_heads * (head_dim / 2);
        let threads = 256;
        let blocks = (num_pairs + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_position = position_buf.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_position) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// CORRECTNESS-011: RoPE NEOX style (split halves)
    ///
    /// Same API as rope_into but uses NEOX-style element pairing (i, i + half_dim)
    /// instead of adjacent pairs (2*i, 2*i+1). Required for Qwen2.5 models.
    pub fn rope_neox_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::RopeNeox {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_neox_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: num_heads blocks, each with half_dim threads
        let config = LaunchConfig::grid_2d(num_heads, 1, head_dim / 2, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut pos_val = position;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
    ///
    /// Same as rope_neox_into but reads position from device memory.
    /// CORRECTNESS-013: When CORRECTNESS_MODE=1, uses PreciseRopeNeoxIndirect kernel
    /// with polynomial sin/cos approximation for CPU-matching precision.
    pub fn rope_neox_indirect_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            let mode = std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!(
                    "[CORRECTNESS-013] RoPE NEOX using PreciseRopeIndirectKernel (polynomial trig)"
                );
            }
            mode
        });

        // Choose kernel type based on mode
        let (kernel_type, cache_key) = if use_precise {
            (
                KernelType::PreciseRopeNeoxIndirect {
                    num_heads,
                    head_dim,
                    theta,
                },
                format!("rope_precise_indirect_{}_{}", num_heads, head_dim),
            )
        } else {
            (
                KernelType::RopeNeoxIndirect {
                    num_heads,
                    head_dim,
                    theta,
                },
                format!("rope_neox_indirect_{}_{}", num_heads, head_dim),
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

        // Grid: num_heads blocks, each with half_dim threads
        let config = LaunchConfig::grid_2d(num_heads, 1, head_dim / 2, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_position = position_buf.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_position) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    // =========================================================================
    // PAR-023: Host Convenience Methods for Activation Kernels
    // =========================================================================

    /// PAR-023: SiLU activation with host memory (convenience)
    ///
    /// Uploads input, runs kernel, syncs, downloads result.
    pub fn silu_host(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), GpuError> {
        let n = input.len() as u32;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let output_gpu = self.silu_gpu(&input_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: GELU activation with host memory (convenience)
    pub fn gelu_host(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), GpuError> {
        let n = input.len() as u32;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let output_gpu = self.gelu_async(&input_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: Element-wise multiply with host memory (convenience)
    pub fn elementwise_mul_host(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = a.len() as u32;
        let a_gpu = GpuBuffer::from_host(&self.context, a)?;
        let b_gpu = GpuBuffer::from_host(&self.context, b)?;
        let output_gpu = self.elementwise_mul_gpu(&a_gpu, &b_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: Fused SwiGLU with host memory (convenience)
    pub fn fused_swiglu_host(
        &mut self,
        gate: &[f32],
        up: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = gate.len() as u32;
        let gate_gpu = GpuBuffer::from_host(&self.context, gate)?;
        let up_gpu = GpuBuffer::from_host(&self.context, up)?;
        let output_gpu = self.fused_swiglu_gpu(&gate_gpu, &up_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-014: Add two GPU buffers element-wise (residual connection)
    ///
    /// Computes: output[i] += input[i] for all i
    /// Uses simple element-wise kernel for residual connections.
    pub fn add_residual_gpu(
        &mut self,
        output: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        // Use BiasActivation kernel with no activation - it adds "bias" to output
        // We repurpose this by treating input as "bias" to add to output
        let kernel_type = KernelType::BiasActivation {
            n,
            bias_size: n,  // Same size as output
            activation: 0, // No activation
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("residual_{}", n);

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

        let mut ptr_output = output.as_ptr();
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

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-014: Q4K GEMV operating on GPU buffers (no CPU round-trip)
    ///
    /// Input and output are GPU-resident buffers. Only weight name lookup uses CPU.
    /// Part of persistent GPU tensor optimization for M4 milestone.
    pub fn q4k_gemv_gpu(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-014: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

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

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-094: Tensor Core Q4K GEMM for batched speculative decode
    ///
    /// Enables M>1 batched forward pass with fused dequant+GEMM using tensor cores.
    /// Target: 8x speedup over GEMV for M≥16 speculative tokens.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP16
    /// * `output` - Output buffer [M, N] in FP16
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_q4k_gemm(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-094: Quantized weight '{}' not cached for GEMM",
                    weight_name
                ))
            })?
            .as_ptr();

        let kernel_type = KernelType::TensorCoreQ4KGemm { m, k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("tc_q4k_gemm_{}_{}_{}", m, k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: ceil(N/16) x ceil(M/16) blocks, 32 threads per block (1 warp for WMMA)
        let grid_x = (n + 15) / 16;
        let grid_y = (m + 15) / 16;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 32, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_output = output.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-095: Tensor Core Q4K GEMM with CPU input/output
    ///
    /// Batched forward pass for speculative decode verification.
    /// Input and output are CPU slices; computation uses GPU-resident Q4K weights.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP32 (converted to FP16 on GPU)
    /// * `output` - Output buffer [M, N] in FP32
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_q4k_gemm_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        // Validate dimensions
        let expected_input = (m as usize) * (k as usize);
        let expected_output = (m as usize) * (n as usize);

        if input.len() != expected_input {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-095: Input size {} != expected M*K = {}*{} = {}",
                input.len(),
                m,
                k,
                expected_input
            )));
        }
        if output.len() != expected_output {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-095: Output size {} != expected M*N = {}*{} = {}",
                output.len(),
                m,
                n,
                expected_output
            )));
        }

        // Get cached weight buffer
        let _weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-095: Quantized weight '{}' not cached for batched GEMM",
                    weight_name
                ))
            })?
            .as_ptr();

        // Upload input to GPU
        let input_buf = GpuBuffer::from_host(&self.context, input)?;
        let output_buf = GpuBuffer::new(&self.context, expected_output)?;

        // Execute kernel
        self.tensor_core_q4k_gemm(weight_name, &input_buf, &output_buf, m, k, n)?;

        // Sync and download output
        self.stream.synchronize()?;
        output_buf.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-096: Batched Q4K GEMV with L2 cache reuse
    ///
    /// Performs M sequential GEMVs using the same cached weights.
    /// Weight data stays in L2 cache between rows, amortizing memory bandwidth.
    /// This enables speculative decode verification without WMMA kernel complexity.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP32
    /// * `output` - Output buffer [M, N] in FP32
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Performance
    /// Expected ~2-3x speedup over M separate calls due to L2 weight caching.
    /// Weights (3MB per layer) fit in RTX 4090 L2 (72MB).
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn batched_q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        // Validate dimensions
        let expected_input = (m as usize) * (k as usize);
        let expected_output = (m as usize) * (n as usize);

        if input.len() != expected_input {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-096: Input size {} != expected M*K = {}*{} = {}",
                input.len(),
                m,
                k,
                expected_input
            )));
        }
        if output.len() != expected_output {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-096: Output size {} != expected M*N = {}*{} = {}",
                output.len(),
                m,
                n,
                expected_output
            )));
        }

        // Get cached weight pointer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-096: Quantized weight '{}' not cached for batched GEMV",
                    weight_name
                ))
            })?
            .as_ptr();

        // PAR-108: Use true batched kernel for dequant sharing across M sequences
        // This amortizes weight dequantization cost across batch, providing ~15x speedup for M=4
        let input_buf = GpuBuffer::from_host(&self.context, input)?;
        let output_buf = GpuBuffer::new(&self.context, output.len())?;

        // Single batched kernel launch - dequantizes once, multiplies by M inputs
        self.batched_q4k_gemv_into(weight_ptr, &input_buf, &output_buf, m, n, k)?;

        // Synchronize and download results
        self.stream.synchronize()?;
        output_buf.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-014: Fused FFN on GPU (up + GELU + down in single GPU round-trip)
    ///
    /// Reduces 2 GPU round-trips to 1 by keeping intermediate FFN hidden state on GPU.
    /// Input and output are CPU slices; intermediate computation stays on GPU.
    ///
    /// # Arguments
    /// * `input` - Hidden state [hidden_dim]
    /// * `output` - Output hidden state [hidden_dim]
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_q4k(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<(), GpuError> {
        // Verify weights are cached
        let up_ptr = self
            .quantized_weight_cache
            .get(ffn_up_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-014: FFN up weight '{}' not cached",
                    ffn_up_name
                ))
            })?
            .as_ptr();

        let down_ptr = self
            .quantized_weight_cache
            .get(ffn_down_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-014: FFN down weight '{}' not cached",
                    ffn_down_name
                ))
            })?
            .as_ptr();

        // 1. Upload input to GPU (only transfer IN for FFN)
        let buf_input = GpuBuffer::from_host(&self.context, input)?;

        // 2. Allocate intermediate buffer for FFN hidden state
        let buf_intermediate = GpuBuffer::<f32>::new(&self.context, intermediate_dim as usize)?;

        // 3. Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, hidden_dim as usize)?;

        // 4. FFN up projection: [hidden_dim] -> [intermediate_dim]
        let up_kernel_type = KernelType::Q4KGemv {
            k: hidden_dim,
            n: intermediate_dim,
        };
        let up_kernel_name = self.kernels.kernel_name(&up_kernel_type);
        let up_cache_key = format!("q4k_gemv_{}_{}", hidden_dim, intermediate_dim);

        if !self.modules.contains_key(&up_cache_key) {
            let ptx = self.kernels.generate_ptx(&up_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(up_cache_key.clone(), module);
        }

        {
            let module = self.modules.get_mut(&up_cache_key).expect("just inserted");
            let config = LaunchConfig::grid_2d(intermediate_dim, 1, 32, 1);

            let mut ptr_output = buf_intermediate.as_ptr();
            let mut ptr_weights = up_ptr;
            let mut ptr_input = buf_input.as_ptr();
            let mut k_val = hidden_dim;
            let mut n_val = intermediate_dim;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    up_kernel_name,
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
        }

        // 5. GELU activation in-place on intermediate buffer
        self.gelu_gpu(&buf_intermediate, intermediate_dim)?;

        // 6. FFN down projection: [intermediate_dim] -> [hidden_dim]
        let down_kernel_type = KernelType::Q4KGemv {
            k: intermediate_dim,
            n: hidden_dim,
        };
        let down_kernel_name = self.kernels.kernel_name(&down_kernel_type);
        let down_cache_key = format!("q4k_gemv_{}_{}", intermediate_dim, hidden_dim);

        if !self.modules.contains_key(&down_cache_key) {
            let ptx = self.kernels.generate_ptx(&down_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(down_cache_key.clone(), module);
        }

        {
            let module = self
                .modules
                .get_mut(&down_cache_key)
                .expect("just inserted");
            let config = LaunchConfig::grid_2d(hidden_dim, 1, 32, 1);

            let mut ptr_output = buf_output.as_ptr();
            let mut ptr_weights = down_ptr;
            let mut ptr_input = buf_intermediate.as_ptr();
            let mut k_val = intermediate_dim;
            let mut n_val = hidden_dim;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    down_kernel_name,
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
        }

        // 7. Sync and download result (only transfer OUT for FFN)
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

}
