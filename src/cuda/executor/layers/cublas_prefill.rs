//! PMAT-024/026/031: cuBLAS GEMM for prefill
//!
//! Prefill path evolution:
//!   v1 (PMAT-024): Dequant Q4K/Q6K → FP32 scratch → cuBLAS SGEMM (per-request dequant)
//!   v2 (PMAT-031): Cache FP16 weights + cuBLAS HGEMM with tensor cores
//!
//! PMAT-031 eliminates per-request dequant cost and uses tensor cores for 2-4x speedup.
//! On first prefill, weights are dequantized to FP32, converted to FP16, and cached.
//! Subsequent prefills use cached FP16 weights directly with HGEMM.

use super::super::*;

/// Minimum M (batch/sequence length) to use cuBLAS GEMM instead of batched GEMV.
/// Below this threshold, batched GEMV is faster due to lower overhead.
///
/// Override with CUBLAS_GEMM_THRESHOLD env var (e.g., =1 for HGEMM decode on high-BW GPUs).
/// Minimum batch size M for cuBLAS SGEMM to beat batched GEMV in decode.
///
/// GH-141 Five-Whys: At M=4, cuBLAS SGEMM dequants Q4K → FP32 (4 B/elem)
/// then runs SGEMM. Batched GEMV reads Q4K directly (0.5625 B/elem) — 7.1x
/// less bandwidth. SGEMM only wins at large M where compute dominates.
///
/// Default=4: cuBLAS SGEMM beats batched GEMV at M=4 (51 vs 35 tok/s).
/// Batched Q4K GEMV at M<=8 uses single warp (32 threads/block) — insufficient
/// parallelism. Multi-warp specializations only exist for M=16/32.
/// TODO: Add M=4 multi-warp kernel, then raise threshold to 8+.
pub(crate) fn cublas_gemm_threshold() -> u32 {
    std::env::var("CUBLAS_GEMM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// PMAT-031: Inline PTX for FP32→FP16 element-wise conversion.
/// Block size 256, one element per thread. Trivially memory-bound (~1μs for 160K elements).
const F32_TO_F16_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_f16(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<4>;
    .reg .f32 %f0;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    cvt.rn.f16.f32 %h0, %f0;
    shl.b64 %rd4, %rd2, 1;
    add.u64 %rd4, %rd0, %rd4;
    st.global.b16 [%rd4], %h0;
L_DONE:
    ret;
}
"#;

impl CudaExecutor {
    /// Initialize cuBLAS handle for prefill GEMM (lazy, called once)
    pub(crate) fn ensure_cublas(&mut self) -> Result<(), GpuError> {
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

    /// PMAT-031: Ensure FP16 activation scratch is large enough
    fn ensure_fp16_activation_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        if self.fp16_activation_scratch_size >= count {
            return Ok(());
        }
        self.fp16_activation_scratch = Some(GpuBuffer::new(&self.context, count)?);
        self.fp16_activation_scratch_size = count;
        Ok(())
    }

    /// PMAT-031: Convert FP32 GPU data to FP16 using inline PTX kernel
    fn convert_f32_to_f16(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        // Compile conversion kernel once
        if !self.modules.contains_key("f32_to_f16") {
            let module = self.compile_ptx(F32_TO_F16_PTX)?;
            self.modules.insert("f32_to_f16".to_string(), module);
        }

        let module = self.modules.get_mut("f32_to_f16").expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: src_ptr and dst_ptr are valid GPU allocations, count verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_f16",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Launch Q4K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q4k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
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

        Ok(())
    }

    /// Launch Q6K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q6k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q6k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q6KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
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

        Ok(())
    }

    /// Dequantize Q4K weights on GPU into FP32 scratch buffer
    fn dequant_q4k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q4k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-026: Dequantize Q6K weights on GPU into FP32 scratch buffer
    fn dequant_q6k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q6k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-031: Get cached FP16 weight or dequant+convert+cache on first access.
    ///
    /// On cache miss: dequant Q4K/Q6K → FP32 scratch → convert to FP16 → cache.
    /// On cache hit: return cached FP16 pointer directly (zero dequant cost).
    fn get_or_cache_fp16_weight(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.fp16_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        // Cache miss: dequant → FP32 scratch
        let f32_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "get_or_cache_fp16_weight: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        // Allocate persistent FP16 buffer [N × K]
        let count = n as usize * k as usize;
        let fp16_buf = GpuBuffer::<u16>::new(&self.context, count)?;
        let fp16_ptr = fp16_buf.as_ptr();

        // Convert FP32 → FP16 (same stream, ordered after dequant)
        self.convert_f32_to_f16(f32_ptr, fp16_ptr, count as u32)?;

        self.fp16_weight_cache.insert(weight_ptr, fp16_buf);
        Ok(fp16_ptr)
    }

    /// PMAT-031: cuBLAS HGEMM prefill — cached FP16 weights + tensor cores
    ///
    /// C[M×N] = Input_fp16[M×K] @ W_fp16[N×K]^T → C is FP32
    ///
    /// Uses gemm_f16_to_f32: FP16 inputs, FP32 output, FP32 accumulation, tensor cores.
    #[allow(clippy::too_many_arguments)]
    fn cublas_prefill_hgemm(
        &mut self,
        w_fp16_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let detail_trace = std::env::var("PREFILL_DETAIL_TRACE").is_ok();
        let t0 = if detail_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Convert FP32 activations → FP16
        let input_count = m as usize * k as usize;
        self.ensure_fp16_activation_scratch(input_count)?;
        let input_fp16_ptr = self
            .fp16_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.convert_f32_to_f16(packed_input_ptr, input_fp16_ptr, input_count as u32)?;

        let t1 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // HGEMM: FP16 weights × FP16 activations → FP32 output (tensor cores)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        let result = handle.gemm_f16_to_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_fp16_ptr,
            k as i32,
            input_fp16_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        );

        if let (Some(t0), Some(t1)) = (t0, t1) {
            self.stream.synchronize()?;
            let t2 = std::time::Instant::now();
            eprintln!(
                "[HGEMM-TRACE] M={} N={} K={}: cvt={:.3}ms cublas={:.3}ms total={:.3}ms",
                m,
                n,
                k,
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t2.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// PMAT-024/026/031/GH-182: cuBLAS GEMM (or fused Q4K GEMM) for prefill
    ///
    /// C[M×N] = Input[M×K] @ W[N×K]^T
    ///
    /// Priority:
    /// 1. FUSED_Q4K_PREFILL=1 + Q4K → tiled fused Q4K GEMM (reads Q4K directly, 3.56x BW savings)
    /// 2. HGEMM_PREFILL!=0 (default) → cached FP16 weights + cuBLAS HGEMM + tensor cores
    /// 3. HGEMM_PREFILL=0 → per-request dequant + cuBLAS SGEMM
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
        // GH-182: Fused Q4K GEMM — reads Q4K directly (0.5625 B/elem vs 2 B/elem HGEMM)
        if qtype == WeightQuantType::Q4K && std::env::var("FUSED_Q4K_PREFILL").as_deref() == Ok("1")
        {
            return self.fused_q4k_gemm_prefill(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        self.ensure_cublas()?;

        // PMAT-031: HGEMM path with cached FP16 weights (default)
        // GH-141: Skip HGEMM when FP16 cache was cleared (batched mode frees it
        // to make room for batched KV caches on 8GB GPUs). Uses SGEMM instead.
        if std::env::var("HGEMM_PREFILL").as_deref() != Ok("0")
            && !self.fp16_weight_cache.is_empty()
        {
            let w_fp16_ptr = self.get_or_cache_fp16_weight(qtype, weight_ptr, n, k)?;
            return self.cublas_prefill_hgemm(
                w_fp16_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // Fallback: dequant + SGEMM (original PMAT-024/026 path)
        let w_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "cublas_prefill_gemm: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
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

    /// GH-182: Fused tiled Q4K GEMM for prefill — reads Q4K weights directly
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// Each thread computes tile_m output rows for one column, loading weight
    /// super-blocks once and reusing across rows. 3.56x bandwidth reduction
    /// vs HGEMM (0.5625 B/elem vs 2 B/elem).
    #[allow(clippy::too_many_arguments)]
    fn fused_q4k_gemm_prefill(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let tile_m: u32 = std::env::var("FUSED_Q4K_TILE_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        let kernel_type = KernelType::QuantizedGemmGgmlTiled { m, n, k, tile_m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemm_ggml_tiled_{m}_{n}_{k}_{tile_m}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/block_threads), ceil(M/tile_m))
        let block_threads = 128u32;
        let grid_x = (n + block_threads - 1) / block_threads;
        let grid_y = (m + tile_m - 1) / tile_m;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, block_threads, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = packed_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

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

        Ok(())
    }

    // ========================================================================
    // PMAT-032: Parallel Prefill Attention via cuBLAS Strided Batched GEMM
    // ========================================================================

    /// PMAT-032: Inline PTX for causal mask + row-wise softmax.
    ///
    /// Grid: (num_heads, M, 1), Block: (32, 1, 1)
    /// Each block processes one (head, token) pair.
    ///
    /// Params: scores_ptr, M, total_len, base_seq_len, num_heads
    /// scores layout: [num_heads, M, total_len] row-major
    ///
    /// For token i (ctaid.y), valid positions are j < base_seq_len + i + 1.
    /// Sets invalid positions to -inf, then computes in-place softmax.
    const CAUSAL_MASK_SOFTMAX_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry causal_mask_softmax(
    .param .u64 param_scores,
    .param .u32 param_m,
    .param .u32 param_total_len,
    .param .u32 param_base_seq_len,
    .param .u32 param_num_heads
) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;

    // head_idx = ctaid.x, token_idx = ctaid.y, tid = tid.x
    ld.param.u64 %rd0, [param_scores];
    ld.param.u32 %r0, [param_m];
    ld.param.u32 %r1, [param_total_len];
    ld.param.u32 %r2, [param_base_seq_len];
    ld.param.u32 %r3, [param_num_heads];

    mov.u32 %r4, %ctaid.x;   // head_idx
    mov.u32 %r5, %ctaid.y;   // token_idx
    mov.u32 %r6, %tid.x;     // lane_id
    mov.u32 %r7, %ntid.x;    // block_size (32)

    // valid_len = base_seq_len + token_idx + 1
    add.u32 %r8, %r2, %r5;
    add.u32 %r8, %r8, 1;

    // row_offset = (head_idx * M + token_idx) * total_len
    mad.lo.u32 %r9, %r4, %r0, %r5;
    mul.lo.u32 %r9, %r9, %r1;

    // row_ptr = scores + row_offset * 4
    cvt.u64.u32 %rd1, %r9;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd1, %rd0, %rd1;

    // === Pass 1: Apply causal mask + find row max ===
    mov.f32 %f0, 0fFF800000;  // max = -inf

    mov.u32 %r10, %r6;  // j = tid
LOOP_MAX:
    setp.ge.u32 %p0, %r10, %r1;  // j >= total_len?
    @%p0 bra DONE_MAX;

    // addr = row_ptr + j * 4
    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    // If j >= valid_len, mask to -inf
    setp.ge.u32 %p1, %r10, %r8;
    @%p1 bra STORE_NEG_INF;

    // Valid position: load and track max
    ld.global.f32 %f1, [%rd2];
    max.f32 %f0, %f0, %f1;
    bra NEXT_MAX;

STORE_NEG_INF:
    mov.f32 %f1, 0fFF800000;
    st.global.f32 [%rd2], %f1;

NEXT_MAX:
    add.u32 %r10, %r10, %r7;  // j += blockDim
    bra LOOP_MAX;

DONE_MAX:
    // Warp reduce max (butterfly)
    shfl.sync.bfly.b32 %f1, %f0, 16, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 8, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 4, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 2, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 1, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    // %f0 = row_max (broadcast to all lanes via shfl)

    // === Pass 2: exp(x - max) and sum ===
    mov.f32 %f2, 0f00000000;  // sum = 0

    mov.u32 %r10, %r6;
LOOP_EXP:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_EXP;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    sub.f32 %f1, %f1, %f0;       // x - max
    // Clamp to prevent overflow: if x-max < -88, set to 0
    mov.f32 %f3, 0fC2B00000;     // -88.0
    setp.lt.f32 %p2, %f1, %f3;
    @%p2 mov.f32 %f1, 0fC2B00000;

    // exp approximation using ex2 (exp2(x * log2(e)))
    mul.f32 %f1, %f1, 0f3FB8AA3B;  // x * log2(e) = x * 1.4426950408889634
    ex2.approx.f32 %f1, %f1;
    st.global.f32 [%rd2], %f1;
    add.f32 %f2, %f2, %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_EXP;

DONE_EXP:
    // Warp reduce sum
    shfl.sync.bfly.b32 %f1, %f2, 16, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 8, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 4, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 2, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 1, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    // %f2 = sum_exp

    // inv_sum = 1.0 / sum_exp
    rcp.approx.f32 %f2, %f2;

    // === Pass 3: normalize ===
    mov.u32 %r10, %r6;
LOOP_NORM:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_NORM;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd2], %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_NORM;

DONE_NORM:
    ret;
}
"#;

    /// PMAT-032: Ensure attention score scratch buffer is large enough
    fn ensure_attn_score_scratch(&mut self, size: usize) -> Result<(), GpuError> {
        if self.prefill_attn_scores_size >= size {
            return Ok(());
        }
        self.prefill_attn_scores = Some(GpuBuffer::new(&self.context, size)?);
        self.prefill_attn_scores_size = size;
        Ok(())
    }

    /// PMAT-032: Bulk scatter M tokens' K/V to cache positions [cache_len..cache_len+M]
    ///
    /// Uses stream-ordered async D2D copies on self.stream, so cuBLAS HGEMM writes
    /// (which also run on self.stream) are guaranteed to complete before the copies
    /// read the K/V buffers. No explicit stream.synchronize() needed.
    fn bulk_scatter_kv(
        &mut self,
        layer_idx: usize,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        m: u32,
        kv_dim: u32,
    ) -> Result<usize, GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let stream_handle = self.stream.raw();

        let new_len = cache_len + m as usize;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-032: KV cache overflow - max_len={max_len}, trying to add {m} at position {cache_len}"
            )));
        }

        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");

        // Scatter K and V for all M tokens using stream-ordered async copies
        for seq_idx in 0..m as usize {
            let position = cache_len + seq_idx;

            // Scatter K
            {
                let k_cache = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: K cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: k_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded,
                    // src_view lifetime managed manually (forget below)
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            k_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        k_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }

            // Scatter V
            {
                let v_cache = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: V cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: v_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            v_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        v_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }
        }

        // Update cache length once for all M tokens
        self.kv_cache_lengths.insert(layer_idx, new_len);

        Ok(cache_len)
    }

    /// PMAT-032: Parallel prefill attention using cuBLAS strided batched GEMM.
    ///
    /// Replaces M sequential `incremental_attention_into_for_capture` calls with:
    /// 1. Bulk KV scatter (M tokens at once)
    /// 2. QK^T via cuBLAS strided batched GEMM (handles GQA)
    /// 3. Causal mask + softmax (single PTX kernel)
    /// 4. Attn × V via cuBLAS strided batched GEMM
    ///
    /// For M=8, 28 layers: 672 kernel launches → 196 (3.4× reduction).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_cublas(
        &mut self,
        layer_idx: usize,
        _q_buf: &GpuBuffer<f32>,
        _k_buf: &GpuBuffer<f32>,
        _v_buf: &GpuBuffer<f32>,
        _attn_out_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Bulk scatter all M tokens' K/V to cache
        // Uses stream-ordered async D2D copies on self.stream — no sync needed
        let cache_len = self.bulk_scatter_kv(layer_idx, k_buf_ptr, v_buf_ptr, m, kv_dim as u32)?;
        let total_len = cache_len + m as usize;

        // 2. Ensure cuBLAS + score scratch buffer
        self.ensure_cublas()?;
        let score_size = num_heads * m as usize * total_len;
        self.ensure_attn_score_scratch(score_size)?;
        let score_ptr = self
            .prefill_attn_scores
            .as_ref()
            .expect("score scratch just allocated")
            .as_ptr();

        // 3. QK^T via cuBLAS strided batched GEMM
        // Score layout (row-major): [num_heads, M, total_len]
        // cuBLAS sees: C_col[total_len, M] = K_col^T[total_len, d] × Q_col[d, M]
        //
        // K cache: [num_kv_heads, max_len, head_dim] row-major
        //   → col-major view: [head_dim, max_len] per kv_head, ld = head_dim
        //   → Trans gives [total_len, head_dim] → m=total_len, k=head_dim
        //
        // Q packed: [M, num_heads, head_dim] row-major
        //   → col-major view per head: [head_dim, M], ld = q_dim
        //   → NoTrans gives [head_dim, M] → k=head_dim, n=M

        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");

        let k_cache_ptr = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
            .as_ptr();
        let v_cache_ptr = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
            .as_ptr();

        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        // For each KV group, launch strided batched GEMM
        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let k_head_ptr =
                k_cache_ptr + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
            let q_head_ptr =
                q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
            let s_head_ptr = score_ptr
                + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::Trans,   // K^T
                trueno_gpu::driver::GemmOp::NoTrans, // Q
                total_len as i32,                    // m (rows of C)
                m as i32,                            // n (cols of C)
                head_dim as i32,                     // k
                scale,                               // alpha = 1/sqrt(d)
                k_head_ptr,                          // A = K cache for this kv_head
                head_dim as i32,                     // lda
                0,                               // stride_a = 0 (shared K for all q heads in group)
                q_head_ptr,                      // B = Q for first head in group
                q_dim as i32,                    // ldb = q_dim (stride between token rows)
                head_dim as i64,                 // stride_b = head_dim (next head)
                0.0,                             // beta
                s_head_ptr,                      // C = score buffer
                total_len as i32,                // ldc
                (m as usize * total_len) as i64, // stride_c = M * total_len (per head)
                heads_per_kv as i32,             // batch = heads per kv group
            )?;
        }

        // 4. Causal mask + softmax (single kernel launch)
        if !self.modules.contains_key("causal_mask_softmax") {
            let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
            self.modules
                .insert("causal_mask_softmax".to_string(), module);
        }

        {
            let module = self
                .modules
                .get_mut("causal_mask_softmax")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (num_heads as u32, m, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };

            let mut s_ptr = score_ptr;
            let mut m_val = m;
            let mut tl_val = total_len as u32;
            let mut base_val = cache_len as u32;
            let mut nh_val = num_heads as u32;

            // SAFETY: score buffer is valid GPU allocation, dimensions verified above
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "causal_mask_softmax",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 5. Attn × V via cuBLAS strided batched GEMM
        // O[h, M, d] = P[h, M, total_len] × V[kv_h, total_len, d]
        // cuBLAS: C_col[d, M] = V_col[d, total_len] × P_col[total_len, M]
        //
        // V cache: [num_kv_heads, max_len, head_dim] row-major
        //   → col-major: [head_dim, max_len], ld = head_dim, NoTrans
        //
        // P (scores after softmax): [num_heads, M, total_len] row-major
        //   → col-major: [total_len, M], ld = total_len, NoTrans
        //
        // O packed: [M, num_heads, head_dim] row-major
        //   → col-major per head: [head_dim, M], ld = q_dim
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let v_head_ptr =
                v_cache_ptr + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
            let p_head_ptr = score_ptr
                + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;
            let o_head_ptr =
                attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::NoTrans, // V (already in correct layout)
                trueno_gpu::driver::GemmOp::NoTrans, // P
                head_dim as i32,                     // m (rows of C = head_dim)
                m as i32,                            // n (cols of C = M)
                total_len as i32,                    // k (inner dim = total_len)
                1.0,                                 // alpha
                v_head_ptr,                          // A = V cache
                head_dim as i32,                     // lda = head_dim
                0,                                   // stride_a = 0 (shared V)
                p_head_ptr,                          // B = P (softmax output)
                total_len as i32,                    // ldb = total_len
                (m as usize * total_len) as i64,     // stride_b = M * total_len
                0.0,                                 // beta
                o_head_ptr,                          // C = output
                q_dim as i32,                        // ldc = q_dim (packed output stride)
                (head_dim) as i64,                   // stride_c = head_dim (next head)
                heads_per_kv as i32,                 // batch
            )?;
        }

        Ok(())
    }

    /// PMAT-024/026: Batched GEMV with cuBLAS GEMM fallback for prefill
    ///
    /// When M >= threshold and weights are Q4K or Q6K, uses cuBLAS GEMM
    /// (HGEMM or SGEMM) instead of batched GEMV.
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
        // GH-141: Batched HW DP4A for Q4K on DP4A-capable GPUs (sm_75+).
        // Reads Q4K weights (0.5625 B/elem) + Q8_1 activations (1.125 B/elem)
        // = 1.69 B/elem total, vs cuBLAS SGEMM 8 B/elem. 4.7x less bandwidth.
        // PMAT-056: Removed !self.is_capturing guard — DP4A kernels are pure GPU
        // kernels (no H2D copies), graph-capturable. Old guard forced FP32 fallback.
        let use_batched_dp4a = qtype == WeightQuantType::Q4K
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling
            && std::env::var("BATCHED_DP4A").as_deref() != Ok("0");

        if use_batched_dp4a {
            return self.batched_hw_dp4a_q4k_gemv_into(
                weight_ptr,
                packed_input,
                packed_output,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // GH-141: Never use cuBLAS during CUDA graph capture. cuBLAS calls
        // are not reliably capturable — they may use internal workspace or
        // stream management that the graph infrastructure cannot replay.
        // Use batched GEMV (custom PTX) instead, which is always capturable.
        let use_cublas = m >= cublas_gemm_threshold()
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0")
            && !self.is_capturing;

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

    /// GH-141: Clear FP16 weight cache to free VRAM for batched KV caches.
    ///
    /// FP16 weights are only needed during HGEMM prefill (not GEMV decode).
    /// Clearing them before batched decode frees ~2944 MB on Qwen2.5 1.5B,
    /// allowing batched KV cache allocation on 8GB GPUs.
    /// The cache is re-warmed lazily on next prefill via get_or_cache_fp16_weight.
    pub(crate) fn clear_fp16_weight_cache(&mut self) {
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        let count = self.fp16_weight_cache.len();
        self.fp16_weight_cache.clear();
        // Also clear FP16 activation scratch (tied to HGEMM path)
        self.fp16_activation_scratch = None;
        self.fp16_activation_scratch_size = 0;
        if count > 0 {
            eprintln!(
                "[GH-141] Cleared FP16 weight cache: {} matrices ({:.1} MB freed)",
                count, cache_mb,
            );
        }
    }

    /// GH-141: Clear prefill CUDA graphs to free VRAM.
    /// Prefill graphs are not used during batched decode and can be recaptured later.
    pub(crate) fn clear_prefill_graphs(&mut self) {
        let count = self.prefill_graphs.len();
        self.prefill_graphs.clear();
        self.prefill_graph_input_buf = None;
        if count > 0 {
            eprintln!("[GH-141] Cleared {} prefill graphs", count);
        }
    }

    /// PMAT-037: Pre-populate FP16 weight cache for HGEMM decode.
    ///
    /// Must be called BEFORE CUDA graph capture, because graph capture doesn't
    /// allow dynamic GPU memory allocation. Also warms up cuBLAS workspace.
    pub(crate) fn warmup_hgemm_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            // Cache all 7 weight matrices per layer
            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim), // Q: [q_dim, hidden]
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim), // K: [kv_dim, hidden]
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim), // V: [kv_dim, hidden]
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim), // O: [hidden, q_dim]
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ), // gate
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim), // up
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ), // down
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0 && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K) {
                    if self.fp16_weight_cache.get(&ptr).is_none() {
                        self.get_or_cache_fp16_weight(qtype, ptr, n, k)?;
                        cached += 1;
                    }
                }
            }
        }

        // Also cache LM head
        if self.lm_head_ptr != 0
            && (self.lm_head_qtype == WeightQuantType::Q4K
                || self.lm_head_qtype == WeightQuantType::Q6K)
        {
            let lm_ptr = self.lm_head_ptr;
            let lm_qtype = self.lm_head_qtype;
            if self.fp16_weight_cache.get(&lm_ptr).is_none() {
                self.get_or_cache_fp16_weight(lm_qtype, lm_ptr, vocab_size, hidden_dim)?;
                cached += 1;
            }
        }

        // Warm up cuBLAS workspace with a tiny GEMM to force internal allocation
        if cached > 0 {
            self.ensure_fp16_activation_scratch(hidden_dim as usize)?;
            let dummy_out = GpuBuffer::<f32>::new(&self.context, hidden_dim as usize)?;
            // Find any cached FP16 weight to run a dummy GEMM
            if let Some((&_ptr, fp16_buf)) = self.fp16_weight_cache.iter().next() {
                let input_fp16_ptr = self
                    .fp16_activation_scratch
                    .as_ref()
                    .expect("scratch just allocated")
                    .as_ptr();
                let handle = self.cublas_handle.as_ref().expect("cublas initialized");
                // Tiny 1×1 GEMM to force cuBLAS workspace allocation
                let _ = handle.gemm_f16_to_f32(
                    trueno_gpu::driver::GemmOp::Trans,
                    trueno_gpu::driver::GemmOp::NoTrans,
                    1,
                    1,
                    1,
                    0.0, // alpha=0 so output doesn't matter
                    fp16_buf.as_ptr(),
                    1,
                    input_fp16_ptr,
                    1,
                    0.0,
                    dummy_out.as_ptr(),
                    1,
                );
            }
            self.stream.synchronize()?;
        }

        let elapsed = start.elapsed();
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        eprintln!(
            "[PMAT-037] FP16 weight cache: {} matrices cached ({:.1} MB) in {:.1}ms",
            cached,
            cache_mb,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }
}
