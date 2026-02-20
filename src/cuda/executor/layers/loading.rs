impl CudaExecutor {

    /// PAR-054-FIX: Pre-load all kernel modules needed for graph capture
    ///
    /// Root cause of CUDA graph capture failure (code 901):
    /// - `CudaModule::from_ptx` calls CUDA driver which allocates memory
    /// - Any memory allocation during graph capture causes error 901
    /// - Solution: Pre-load ALL modules before `begin_capture()`
    ///
    /// Five-Whys Analysis:
    /// 1. Why does capture fail? Memory allocation detected during capture
    /// 2. Why allocation during capture? Lazy module loading in kernel dispatch
    /// 3. Why lazy loading? Performance optimization for unused kernels
    /// 4. Why does lazy loading allocate? PTX compilation requires driver memory
    /// 5. Why not pre-loaded? Missing pre-loading step before capture
    #[allow(clippy::too_many_lines)]
    pub(crate) fn preload_modules_for_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let max_len = self.kv_cache_max_len as u32;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // 1. RMSNorm kernel (used for attn_norm, ffn_norm, output_norm)
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if use_precise {
            // CORRECTNESS-013: Preload PreciseRmsNorm for CPU-matching precision
            let rmsnorm_key = format!("rmsnorm_precise_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::PreciseRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        } else {
            // PAR-081: Use VectorizedRmsNorm with 256 threads (8x faster than single-warp)
            let rmsnorm_key = format!("rmsnorm_vectorized_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::VectorizedRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5, // Runtime parameter, kernel code same regardless
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        }

        // 2. Q/K/V GEMV kernels - pre-load all quant types that might be used
        // PAR-082-V3: MwvQ4KGemv with configurable warp count
        let nw = crate::cuda::kernels::mwv_warp_count();
        let mwv_q4k_q_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, q_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_q_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim,
                n: q_dim,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_q_key, module);
        }
        let mwv_q4k_kv_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, kv_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_kv_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim,
                n: kv_dim,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_kv_key, module);
        }

        // Q5_0 GEMV (for Qwen 0.5B which uses Q5_0 for Q/K)
        let q5_0_q_key = format!("q5_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q5_0_q_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q5_0_q_key, module);
        }
        let q5_0_kv_key = format!("q5_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q5_0_kv_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q5_0_kv_key, module);
        }

        // Q6K GEMV for Q projection - PAR-066: Preload both original and coalesced versions
        // Original Q6K (for non-256-aligned K dimensions)
        let q6k_q_key = format!("q6k_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q6k_q_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q6k_q_key, module);
        }
        // PAR-066: CoalescedQ6K for Q (byte-wise scale loading, fixes alignment issue)
        if hidden_dim.is_multiple_of(256) {
            let coalesced_q6k_q_key = format!("coalesced_q6k_gemv_{}_{}", hidden_dim, q_dim);
            if !self.modules.contains_key(&coalesced_q6k_q_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: q_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(coalesced_q6k_q_key, module);
            }
        }
        // Q6K GEMV for KV projection
        let q6k_kv_key = format!("q6k_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q6k_kv_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q6k_kv_key, module);
        }
        // PAR-066: CoalescedQ6K for KV
        if hidden_dim.is_multiple_of(256) {
            let coalesced_q6k_kv_key = format!("coalesced_q6k_gemv_{}_{}", hidden_dim, kv_dim);
            if !self.modules.contains_key(&coalesced_q6k_kv_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: kv_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(coalesced_q6k_kv_key, module);
            }
        }

        // Q8_0 GEMV
        let q8_0_q_key = format!("q8_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q8_0_q_key) {
            let kernel_type = KernelType::Q8_0Gemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q8_0_q_key, module);
        }
        let q8_0_kv_key = format!("q8_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q8_0_kv_key) {
            let kernel_type = KernelType::Q8_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q8_0_kv_key, module);
        }

        // 3. Output projection (q_dim -> hidden_dim) - PAR-082-V3: MWV Q4K
        let mwv_q4k_o_key = format!("mwv_q4k_gemv_{}_{}_{}", q_dim, hidden_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_o_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: q_dim,
                n: hidden_dim,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_o_key, module);
        }

        // 4. FFN GEMV kernels (gate/up: hidden->intermediate, down: intermediate->hidden)
        let mwv_q4k_up_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, intermediate_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_up_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim,
                n: intermediate_dim,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_up_key, module);
        }
        let mwv_q4k_down_key = format!("mwv_q4k_gemv_{}_{}_{}", intermediate_dim, hidden_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_down_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: intermediate_dim,
                n: hidden_dim,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_down_key, module);
        }

        // Q6K FFN down (some models use Q6K for FFN down)
        let q6k_down_key = format!("q6k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q6k_down_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q6k_down_key, module);
        }
        // PAR-066: CoalescedQ6K for FFN down (byte-wise scale loading)
        if intermediate_dim.is_multiple_of(256) {
            let coalesced_q6k_down_key =
                format!("coalesced_q6k_gemv_{}_{}", intermediate_dim, hidden_dim);
            if !self.modules.contains_key(&coalesced_q6k_down_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: intermediate_dim,
                    n: hidden_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(coalesced_q6k_down_key, module);
            }
        }

        // 5-9. LM head, RoPE, SwiGLU, residual, scatter, attention kernels
        self.preload_lm_head_and_utility_modules(
            num_layers, hidden_dim, intermediate_dim, vocab_size,
            num_heads, num_kv_heads, head_dim, max_len, q_dim, kv_dim, nw,
        )
    }
}

include!("preload_utilities.rs");
