//! PMAT-024/026: cuBLAS GEMM for prefill
//!
//! Replaces batched GEMV (tiles M=8, re-reads weights per tile) with:
//!   1. Dequantize Q4K/Q6K weights → dense FP32 scratch buffer (GPU kernel)
//!   2. cuBLAS SGEMM: C[M×N] = A[M×K] @ B[K×N]  (reads weights ONCE)
//!
//! Expected improvement: 86x prefill gap → ~2-5x (cuBLAS vs llama.cpp cuBLAS).

use super::super::*;

/// Minimum M (batch/sequence length) to use cuBLAS GEMM instead of batched GEMV.
/// Below this threshold, batched GEMV is faster due to lower overhead.
const CUBLAS_PREFILL_THRESHOLD: u32 = 4;

impl CudaExecutor {
    /// Initialize cuBLAS handle for prefill GEMM (lazy, called once)
    fn ensure_cublas(&mut self) -> Result<(), GpuError> {
        if self.cublas_handle.is_some() {
            return Ok(());
        }
        let handle = trueno_gpu::driver::CublasHandle::new(&self.context)?;
        handle.set_stream(&self.stream)?;
        self.cublas_handle = Some(handle);
        Ok(())
    }

    /// Ensure dequant scratch buffer is large enough for N×K FP32 elements
    fn ensure_dequant_scratch(&mut self, n: u32, k: u32) -> Result<(), GpuError> {
        let needed = n as usize * k as usize;
        if self.dequant_scratch_size >= needed {
            return Ok(());
        }
        self.dequant_scratch = Some(GpuBuffer::new(&self.context, needed)?);
        self.dequant_scratch_size = needed;
        Ok(())
    }

    /// Dequantize Q4K weights on GPU into FP32 scratch buffer
    fn dequant_q4k_to_scratch(
        &mut self,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;

        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        // Load dequant kernel module
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self.modules.get_mut(&cache_key).expect("module just inserted");

        // Grid: (N, num_sb) — one block per super-block per row
        // Block: 32 threads (one warp)
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = scratch_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q4k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(scratch_ptr)
    }

    /// PMAT-026: Dequantize Q6K weights on GPU into FP32 scratch buffer
    fn dequant_q6k_to_scratch(
        &mut self,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;

        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        let num_sb = (k + 255) / 256;
        let cache_key = format!("q6k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q6KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self.modules.get_mut(&cache_key).expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = scratch_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q6k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(scratch_ptr)
    }

    /// PMAT-024/026: cuBLAS GEMM for prefill — dequant → FP32 → cuBLAS SGEMM
    ///
    /// C[M×N] = A[M×K] @ W_dequant[N×K]^T
    ///
    /// Dispatches dequant kernel by weight type (Q4K or Q6K), then calls cuBLAS.
    /// Row-major C[M,N] = Input[M,K] @ W[N,K]^T
    /// cuBLAS: gemm(Trans, NoTrans, N, M, K, 1.0, w_ptr, K, input, K, 0.0, output, N)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cublas_prefill_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        self.ensure_cublas()?;

        // Step 1: Dequantize weights to FP32 scratch [N × K]
        let w_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => return Err(GpuError::InvalidParameter(format!(
                "cublas_prefill_gemm: unsupported qtype {:?}", qtype
            ))),
        };

        // Step 2: cuBLAS SGEMM
        // W[N,K] row-major → col-major [K,N] ld=K → Trans gives [N,K]
        // Input[M,K] row-major → col-major [K,M] ld=K → NoTrans gives [K,M]
        // C'[N,M] = W[N,K] @ Input^T[K,M]
        let handle = self.cublas_handle.as_ref().expect("cublas just initialized");
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_ptr,
            k as i32,
            packed_input_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        )
    }

    /// PMAT-024/026: Batched GEMV with cuBLAS GEMM fallback for prefill
    ///
    /// When M >= threshold and weights are Q4K or Q6K, uses cuBLAS GEMM
    /// (dequant + SGEMM) instead of batched GEMV.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_gemv_or_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input: &GpuBuffer<f32>,
        packed_output: &GpuBuffer<f32>,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n_per_seq: u32,
        k_per_seq: u32,
    ) -> Result<(), GpuError> {
        let use_cublas = m >= CUBLAS_PREFILL_THRESHOLD
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0");

        if use_cublas {
            self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        } else {
            self.batched_gemv_with_fallback(
                qtype,
                weight_ptr,
                packed_input,
                packed_output,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        }
    }
}
