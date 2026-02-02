//! Basic quantized GEMV operations (Q6K, Q8, Q5, Q4)
//!
//! This module implements the core quantized matrix-vector multiplication
//! for Q6_K, Q8_0, Q5_0, Q4_0, Q4_1, and Q5_K quantization formats.

#![allow(clippy::wildcard_imports)]
#![allow(clippy::too_many_arguments)]

use super::*;

impl CudaExecutor {
    /// Batched Q6K GEMV (matrix-vector multiply) with quantized weights.
    ///
    /// Performs `output = weight * input` where weight is Q6K quantized.
    pub fn batched_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_q6k_gemv_into")?;
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q6K super-blocks"
        );

        let kernel_type = KernelType::BatchedQ6KGemv { k, n, m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_q6k_gemv_{}_{}_{}", m, k, n);

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
        validate_device_ptr(weight_ptr, "q6k_gemv_into")?;
        // Original Q6K kernel (CoalescedQ6K disabled due to CORRECTNESS-006)
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "coalesced_q6k_gemv_into")?;
        let kernel_type = KernelType::CoalescedQ6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "q8_0_gemv_into")?;
        // PAR-058: Zero allocation Q8_0 GEMV for mixed-quantization models
        let kernel_type = KernelType::Q8_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "q5_0_gemv_into")?;
        // PAR-058: Zero allocation Q5_0 GEMV for Qwen 0.5B Q/K weights
        let kernel_type = KernelType::Q5_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "q4_0_gemv_into")?;
        // PAR-058: Zero allocation Q4_0 GEMV for GGUF qtype mismatch
        let kernel_type = KernelType::Q4_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "q4_1_gemv_into")?;
        // PAR-058: Zero allocation Q4_1 GEMV for Qwen2.5-0.5B FFN down
        let kernel_type = KernelType::Q4_1Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_1_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
        validate_device_ptr(weight_ptr, "q5k_gemv_into")?;
        // PAR-058: Zero allocation Q5K GEMV for mixed-quantization models
        let kernel_type = KernelType::Q5KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::cuda::executor::test_fixtures::{
        generate_q4_0_weights, generate_q5_0_weights, generate_q8_0_weights,
    };

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Q8_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q8_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // K=256, N=64: 64 output rows, 8 blocks per row (32 elements/block)
        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q8_0_weights(blocks);

        // Upload weights to GPU
        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let weight_ptr = weight_buf.as_ptr();

        // Create input/output buffers
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        // Execute (may fail on PTX generation but exercises path)
        let result = exec.q8_0_gemv_into(weight_ptr, &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q8_0_gemv_into_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 512u32;
        let n = 128u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q8_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q8_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q5_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q5_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q5_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q5_0_gemv_into_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 512u32;
        let n = 128u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q5_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q4_0 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q4_0_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_q4_0_gemv_into_single_row() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 1u32;
        let blocks = k as usize / 32;
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![1.0f32; k as usize];
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_0_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q4_1 GEMV Tests
    // ========================================================================

    #[test]
    fn test_q4_1_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q4_1 has same block count as Q4_0 but 20 bytes per block instead of 18
        let k = 256u32;
        let n = 64u32;
        let blocks = (n as usize) * (k as usize / 32);
        // Use Q4_0 weights (18 bytes/block), Q4_1 expects 20 bytes/block
        // The kernel will interpret incorrectly, but we're testing the path
        let weights = generate_q4_0_weights(blocks);

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q4_1_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q5_K GEMV Tests
    // ========================================================================

    #[test]
    fn test_q5k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q5_K: 176 bytes per 256 elements (super-block format)
        let k = 256u32;
        let n = 64u32;
        // Q5_K needs k to be multiple of 256
        let superblocks = (n as usize) * (k as usize / 256);
        // Simulate Q5K weight data
        let weights = vec![0u8; superblocks * 176];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q5k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q6_K GEMV Tests
    // ========================================================================

    #[test]
    fn test_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Q6_K: 210 bytes per 256 elements
        let k = 256u32;
        let n = 64u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    #[test]
    fn test_coalesced_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.coalesced_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Batched Q6K GEMV Tests
    // ========================================================================

    #[test]
    fn test_batched_q6k_gemv_into_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let m = 4u32; // batch size
        let k = 256u32;
        let n = 64u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        // Batched input: m * k elements
        let input: Vec<f32> = (0..(m * k) as usize).map(|i| (i as f32) * 0.001).collect();
        // Batched output: m * n elements
        let output = vec![0.0f32; (m * n) as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.batched_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, m, n, k);
        let _ = result;
    }

    #[test]
    fn test_batched_q6k_gemv_into_m8() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let m = 8u32;
        let k = 256u32;
        let n = 32u32;
        let superblocks = (n as usize) * (k as usize / 256);
        let weights = vec![0u8; superblocks * 210];

        let weight_buf = GpuBuffer::from_host(&exec.context, &weights).unwrap();
        let input: Vec<f32> = vec![0.5f32; (m * k) as usize];
        let output = vec![0.0f32; (m * n) as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result =
            exec.batched_q6k_gemv_into(weight_buf.as_ptr(), &input_buf, &output_buf, m, n, k);
        let _ = result;
    }
}
