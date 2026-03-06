//! PMAT-024: cuBLAS GEMM for prefill
//!
//! Replaces batched GEMV (tiles M=8, re-reads weights per tile) with:
//!   1. Dequantize Q4K weights → dense FP32 scratch buffer (GPU kernel)
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

    /// PMAT-024: cuBLAS GEMM for prefill — dequant Q4K → FP32 → cuBLAS SGEMM
    ///
    /// C[M×N] = A[M×K] @ W_dequant[K×N]^T
    ///
    /// Where:
    ///   A = packed input [M × K] (row-major FP32)
    ///   W_dequant = dequantized weights [N × K] (row-major FP32, Q4K source)
    ///   C = output [M × N] (row-major FP32)
    ///
    /// cuBLAS column-major trick:
    ///   C_row = A_row @ W_row^T  →  C_col^T = W_col @ A_col^T
    ///   = cuBLAS(NoTrans, Trans, N, M, K, W, A, C)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cublas_prefill_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        self.ensure_cublas()?;

        // Step 1: Dequantize Q4K weights to FP32 scratch [N × K]
        let w_ptr = self.dequant_q4k_to_scratch(weight_ptr, n, k)?;

        // Step 2: cuBLAS SGEMM
        // Row-major: C[M,N] = A[M,K] @ W[N,K]^T
        // cuBLAS col-major: C^T = W @ A^T
        //   transa=NoTrans (W is [N×K] col-major = [K×N]^T = we need W as-is)
        //   Actually for row-major → col-major:
        //   C_row[M,N] = A_row[M,K] @ W_row[N,K]^T
        //   ↔ C_col[N,M] = W_col[N,K] @ A_col[K,M]
        //   ↔ cuBLAS(NoTrans, NoTrans, N, M, K, 1.0, W, N, A, K, 0.0, C, N)
        //
        //   But W is stored row-major [N,K], cuBLAS sees col-major [K,N] = W^T
        //   So we need Trans on W: cuBLAS(Trans, NoTrans, N, M, K, W, K, A, K, C, N)
        //   Wait — let me use the identity correctly.
        //
        //   Row-major C[M,N] = A[M,K] @ B[K,N]  where B = W^T
        //   But W is [N,K] row-major. W^T is [K,N].
        //   So B = W^T is [K,N] row-major = exactly what we need for A @ B.
        //
        //   Using gemm_f32_row_major(m, n, k, A, B, C) where B[K,N]:
        //   But we have W[N,K], not B[K,N].
        //
        //   cuBLAS column-major: C_col = B_col @ A_col
        //   For row-major C[M,N] = A[M,K] @ W[N,K]^T:
        //   Treat as col-major: C_col[N,M] = (W^T)_col @ A_col
        //   W is [N,K] row-major = [K,N] col-major
        //   (W^T)_col = W_row viewed as col-major... this IS [K,N] col-major = W^T in col-major
        //   We need (W^T)_col which is just W_row interpreted as col-major = [K,N]
        //   Actually W[N,K] row-major in memory = W[K,N] col-major = W^T in col-major
        //   So cuBLAS NoTrans on W gives us W^T (what we want)
        //
        //   cuBLAS(NoTrans, NoTrans, N, M, K, 1.0, w_ptr, N, input, K, 0.0, output, N)
        //   But... wait. W[N,K] row-major: element [i,j] at offset i*K+j
        //   Interpreted as col-major [K,N]: element [j,i] at offset i*K+j → lda = K
        //   So W col-major is [K,N] with lda=K.
        //   NoTrans means op(W) = W_col = [K,N]
        //   We need N,M result. op(W)[K,N] @ op(A)[?,M] → need K rows from A
        //   A[M,K] row-major = [K,M] col-major with lda=K.
        //   NoTrans op(A) = [K,M].
        //   Result: [K,N]^T... no, GEMM: C = op(A) @ op(B) where C is [m,n]
        //
        //   Let me just be direct:
        //   cuBLAS: C[m_cb, n_cb] = alpha * op(A)[m_cb, k_cb] @ op(B)[k_cb, n_cb] + beta * C
        //   All col-major. We want row-major C[M,N] = Input[M,K] @ W[N,K]^T
        //
        //   Trick: row-major C[M,N] stored = col-major C'[N,M]
        //   C'[N,M] = (Input @ W^T)^T = W @ Input^T
        //   Input[M,K] row-major stored = col-major Input'[K,M] with ld=K
        //   W[N,K] row-major stored = col-major W'[K,N] with ld=K
        //
        //   C'[N,M] = W @ Input^T
        //   In col-major world: W'[K,N] but we need W[N,K] col-major...
        //   W[N,K] row-major = memory layout: row i has K elements = col-major [K,N] ld=K
        //   So W_col = [K,N] ld=K. op(W) with Trans = [K,N]^T = [N,K]. That gives us W.
        //   op(W) NoTrans = [K,N]. That gives W^T.
        //
        //   C'[N,M] = W @ Input^T
        //   = op_a(W_mem) @ op_b(Input_mem)
        //   W_mem is [K,N] col-major. We need W[N,K]. So op_a = Trans → [N,K]... no wait.
        //   C'[N,M] = W[N,K] @ Input^T[K,M]
        //   W_mem col-major = [K,N] ld=K → Trans gives [N,K] ✓
        //   Input_mem col-major = [K,M] ld=K → NoTrans gives [K,M] ✓
        //
        //   cuBLAS call: gemm(Trans, NoTrans, N, M, K, 1.0, w_ptr, K, input, K, 0.0, output, N)

        let handle = self.cublas_handle.as_ref().expect("cublas just initialized");
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,    // W[N,K] stored row-major → col-major [K,N] → Trans gives [N,K]
            trueno_gpu::driver::GemmOp::NoTrans,  // Input[M,K] stored row-major → col-major [K,M] → NoTrans gives [K,M]
            n as i32,                              // m_cublas = N (rows of result in col-major)
            m as i32,                              // n_cublas = M (cols of result in col-major)
            k as i32,
            1.0,
            w_ptr,
            k as i32,                              // lda = K (W stored row-major [N,K])
            packed_input_ptr,
            k as i32,                              // ldb = K (Input stored row-major [M,K])
            0.0,
            packed_output_ptr,
            n as i32,                              // ldc = N (Output stored row-major [M,N])
        )
    }

    /// PMAT-024: Batched GEMV with cuBLAS GEMM fallback for prefill
    ///
    /// When M > CUBLAS_PREFILL_THRESHOLD and weights are Q4K, uses cuBLAS GEMM
    /// (dequant + SGEMM) instead of batched GEMV. This reads weights once instead
    /// of M/8 times, closing the 86x prefill gap.
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
        // Use cuBLAS GEMM for Q4K when M exceeds threshold and CUBLAS_PREFILL env is not "0"
        let use_cublas = m >= CUBLAS_PREFILL_THRESHOLD
            && qtype == WeightQuantType::Q4K
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0");

        if use_cublas {
            self.cublas_prefill_gemm(
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
