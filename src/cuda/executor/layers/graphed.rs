//! CUDA Graph-captured forward pass operations
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-054: forward_all_layers_gpu_to_logits_graphed
//! - PAR-062: gpu_argmax
//! - PAR-062: forward_graphed_replay_to_token_id

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// PAR-054: Graph-captured forward pass for decode (M=1)
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead from ~280 launches
    /// to 1 graph launch (~10µs vs ~5.6ms overhead).
    ///
    /// First decode token: captures the kernel sequence into a graph
    /// Subsequent tokens: replays the captured graph with updated position
    ///
    /// # Performance
    ///
    /// - Without graphs: ~280 kernel launches × ~20µs = ~5.6ms overhead/token
    /// - With graphs: 1 graph launch × ~10µs = ~0.01ms overhead/token
    /// - Expected speedup: ~500x reduction in launch overhead
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits_graphed(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // PAR-054: Environment variable to disable CUDA graphs for debugging
        // Set CUDA_GRAPH_DISABLE=1 to use non-graphed path
        static GRAPH_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graph_disabled = *GRAPH_DISABLED.get_or_init(|| {
            std::env::var("CUDA_GRAPH_DISABLE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if graph_disabled {
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-064-DEBUG: Allow disabling graph mode for debugging
        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);

        // PAR-054: Check if we should capture or replay
        if !skip_graph && self.decode_graph.is_some() && self.decode_token_count > 0 {
            // Replay path: update position and launch graph
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return self.forward_graphed_replay(input, logits, position);
        }

        // First token or no graph yet: try to capture
        // We need workspace path for stable addresses
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            // Fall back to non-graphed path if workspace not available
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path (has_workspace={}, has_indexed={}, layers={})",
                self.has_workspace(), self.has_indexed_weights(), self.indexed_layer_weights.len());
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Verify lm_head_ptr is set (needed for graph-captured LM head projection)
        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Initialize position buffer if needed
        if self.position_buf.is_none() {
            let pos_buf = GpuBuffer::from_host(&self.context, &[position])?;
            self.position_buf = Some(pos_buf);
        } else {
            // Update position
            self.position_buf
                .as_mut()
                .expect("position_buf must be initialized")
                .copy_from_host(&[position])?;
        }

        // PAR-061: Initialize seq_len buffer for indirect attention kernel
        // seq_len = position + 1 (new sequence length after adding this token)
        let seq_len = position + 1;
        if self.seq_len_buf.is_none() {
            let len_buf = GpuBuffer::from_host(&self.context, &[seq_len])?;
            self.seq_len_buf = Some(len_buf);
        } else {
            self.seq_len_buf
                .as_mut()
                .expect("seq_len_buf must be initialized")
                .copy_from_host(&[seq_len])?;
        }

        // PAR-054: Initialize stable input buffer if needed
        let hidden_size = hidden_dim as usize;
        if self.graph_input_buf.is_none()
            || self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len()
                != hidden_size
        {
            let input_buf = GpuBuffer::from_host(&self.context, input)?;
            self.graph_input_buf = Some(input_buf);
        } else {
            self.graph_input_buf
                .as_mut()
                .expect("graph_input_buf must be initialized")
                .copy_from_host(input)?;
        }

        // PAR-054: Pre-allocate normed_hidden_buf before capture
        if self.workspace.normed_hidden_buf.is_none() {
            let normed_buf = GpuBuffer::new(&self.context, hidden_size)?;
            self.workspace.normed_hidden_buf = Some(normed_buf);
        }

        // PAR-054: Pre-allocate logits_buf before capture
        if self.workspace.logits_buf.is_none() {
            let logits_buf = GpuBuffer::new(&self.context, vocab_size as usize)?;
            self.workspace.logits_buf = Some(logits_buf);
        }

        // PAR-054-FIX: Pre-load all kernel modules BEFORE graph capture
        // Root cause: CudaModule::from_ptx allocates memory which breaks capture
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // PAR-064-DEBUG: Skip graph capture if SKIP_CUDA_GRAPH=1
        if skip_graph {
            eprintln!("[PAR-064-DEBUG] SKIP_CUDA_GRAPH=1, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Try CUDA graph capture, fall back to non-graphed path if fails
        // Some operations (memory allocation, synchronization) aren't graph-compatible
        let capture_result = self.try_graph_capture(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        match capture_result {
            Ok(()) => {
                // CORRECTNESS-010: Graph capture defines the work but doesn't execute it.
                // Must launch the graph once to produce actual output for first token.
                if let Some(ref graph_exec) = self.decode_graph {
                    self.stream.launch_graph(graph_exec)?;
                }
                // Graph captured and launched, sync and download logits
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(e) => {
                // Graph capture failed, fall back to non-graphed path
                // This is expected for complex operations with allocations
                eprintln!(
                    "[PAR-054] Graph capture failed: {:?}, using non-graphed path",
                    e
                );
                // PAR-070: Pass position for correct RoPE and KV cache behavior
                self.forward_all_layers_gpu_to_logits(
                    input,
                    logits,
                    position,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    vocab_size,
                    epsilon,
                )
            },
        }
    }

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
    fn preload_modules_for_capture(
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
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        }

        // 2. Q/K/V GEMV kernels - pre-load all quant types that might be used
        // PAR-065: Use Coalesced Q4K kernels for better bandwidth (vectorized loads)

        // PAR-065: Coalesced Q4K GEMV for Q (hidden_dim -> q_dim)
        // Five-Whys root cause: TiledQ4KGemv uses single-byte loads (6% bandwidth)
        // CoalescedQ4KGemv uses vectorized u32 loads + warp shuffles (27% speedup)
        let q4k_q_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q4k_q_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_q_key, module);
        }

        // PAR-065: Coalesced Q4K GEMV for K/V (hidden_dim -> kv_dim)
        let q4k_kv_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q4k_kv_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_kv_key, module);
        }

        // Q5_0 GEMV (for Qwen 0.5B which uses Q5_0 for Q/K)
        let q5_0_q_key = format!("q5_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q5_0_q_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q5_0_q_key, module);
        }
        let q5_0_kv_key = format!("q5_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q5_0_kv_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q8_0_q_key, module);
        }
        let q8_0_kv_key = format!("q8_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q8_0_kv_key) {
            let kernel_type = KernelType::Q8_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q8_0_kv_key, module);
        }

        // 3. Output projection (q_dim -> hidden_dim) - PAR-065: Coalesced Q4K
        let q4k_o_key = format!("coalesced_q4k_gemv_{}_{}", q_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_o_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: q_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_o_key, module);
        }

        // 4. FFN GEMV kernels (gate/up: hidden->intermediate, down: intermediate->hidden)
        // PAR-065: Coalesced Q4K for FFN gate/up
        let q4k_up_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, intermediate_dim);
        if !self.modules.contains_key(&q4k_up_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: intermediate_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_up_key, module);
        }
        // PAR-065: Coalesced Q4K for FFN down (K=intermediate_dim)
        // CoalescedQ4KGemv doesn't have the shared memory limitation of TiledQ4KGemv
        let q4k_down_key = format!("coalesced_q4k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_down_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_down_key, module);
        }
        // Pre-load basic Q4K as fallback for non-256-aligned dimensions
        let q4k_down_fallback_key = format!("q4k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_down_fallback_key) {
            let kernel_type = KernelType::Q4KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_down_fallback_key, module);
        }

        // Q6K FFN down (some models use Q6K for FFN down)
        let q6k_down_key = format!("q6k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q6k_down_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
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
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_q6k_down_key, module);
            }
        }

        // 5. LM head (hidden_dim -> vocab_size) - pre-load both Q4K and Q6K
        // PAR-058: Qwen 1.5B uses Q6K for LM head, not Q4K
        // PAR-065: Coalesced Q4K for LM head
        let lm_head_q4k_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, vocab_size);
        if !self.modules.contains_key(&lm_head_q4k_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: vocab_size,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(lm_head_q4k_key, module);
        }
        // Q6K LM head (Qwen 1.5B uses this)
        let lm_head_q6k_key = format!("q6k_gemv_{}_{}", hidden_dim, vocab_size);
        if !self.modules.contains_key(&lm_head_q6k_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: vocab_size,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(lm_head_q6k_key, module);
        }
        // PAR-066: CoalescedQ6K for LM head
        if hidden_dim.is_multiple_of(256) {
            let coalesced_lm_head_q6k_key =
                format!("coalesced_q6k_gemv_{}_{}", hidden_dim, vocab_size);
            if !self.modules.contains_key(&coalesced_lm_head_q6k_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: vocab_size,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_lm_head_q6k_key, module);
            }
        }

        // 6. RoPE kernels (for Q and K)
        // Note: theta is a runtime parameter, cache key only uses num_heads and head_dim
        let theta = self.rope_theta;
        let rope_q_key = format!("rope_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_key) {
            let kernel_type = KernelType::Rope {
                num_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_q_key, module);
        }
        let rope_k_key = format!("rope_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_key) {
            let kernel_type = KernelType::Rope {
                num_heads: num_kv_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_k_key, module);
        }

        // CORRECTNESS-010: RoPE indirect kernels for CUDA graph capture
        // These read position from device memory instead of kernel parameter
        let rope_q_indirect_key = format!("rope_indirect_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_indirect_key) {
            let kernel_type = KernelType::RopeIndirect {
                num_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_q_indirect_key, module);
        }
        let rope_k_indirect_key = format!("rope_indirect_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_indirect_key) {
            let kernel_type = KernelType::RopeIndirect {
                num_heads: num_kv_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_k_indirect_key, module);
        }

        // CORRECTNESS-011: RoPE NEOX indirect kernels for Qwen2.5 (rope_type=2)
        // Five-Whys: GPU garbage output → wrong RoPE style → NEOX kernels not loaded
        // CORRECTNESS-013: Use precise kernels when CORRECTNESS_MODE=1
        if self.rope_type == 2 {
            if use_precise {
                // CORRECTNESS-013: Preload PreciseRopeNeoxIndirect for Q
                let rope_precise_q_indirect_key =
                    format!("rope_precise_indirect_{}_{}", num_heads, head_dim);
                if !self.modules.contains_key(&rope_precise_q_indirect_key) {
                    let kernel_type = KernelType::PreciseRopeNeoxIndirect {
                        num_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_precise_q_indirect_key, module);
                }
                // CORRECTNESS-013: Preload PreciseRopeNeoxIndirect for K
                let rope_precise_k_indirect_key =
                    format!("rope_precise_indirect_{}_{}", num_kv_heads, head_dim);
                if !self.modules.contains_key(&rope_precise_k_indirect_key) {
                    let kernel_type = KernelType::PreciseRopeNeoxIndirect {
                        num_heads: num_kv_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_precise_k_indirect_key, module);
                }
            } else {
                // Standard RopeNeoxIndirect for Q
                let rope_neox_q_indirect_key =
                    format!("rope_neox_indirect_{}_{}", num_heads, head_dim);
                if !self.modules.contains_key(&rope_neox_q_indirect_key) {
                    let kernel_type = KernelType::RopeNeoxIndirect {
                        num_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_neox_q_indirect_key, module);
                }
                // Standard RopeNeoxIndirect for K
                let rope_neox_k_indirect_key =
                    format!("rope_neox_indirect_{}_{}", num_kv_heads, head_dim);
                if !self.modules.contains_key(&rope_neox_k_indirect_key) {
                    let kernel_type = KernelType::RopeNeoxIndirect {
                        num_heads: num_kv_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_neox_k_indirect_key, module);
                }
            }
            // Also preload direct NEOX kernels for non-graph-capture mode
            let rope_neox_q_key = format!("rope_neox_{}_{}", num_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_q_key) {
                let kernel_type = KernelType::RopeNeox {
                    num_heads,
                    head_dim,
                    theta,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rope_neox_q_key, module);
            }
            let rope_neox_k_key = format!("rope_neox_{}_{}", num_kv_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_k_key) {
                let kernel_type = KernelType::RopeNeox {
                    num_heads: num_kv_heads,
                    head_dim,
                    theta,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rope_neox_k_key, module);
            }
        }

        // 7. SwiGLU kernel
        let swiglu_key = format!("fused_swiglu_{}", intermediate_dim);
        if !self.modules.contains_key(&swiglu_key) {
            let kernel_type = KernelType::FusedSwiglu {
                n: intermediate_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(swiglu_key, module);
        }

        // 8. Residual add kernel
        let residual_key = format!("residual_add_{}", hidden_dim);
        if !self.modules.contains_key(&residual_key) {
            let kernel_type = KernelType::ResidualAdd { n: hidden_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(residual_key, module);
        }

        // 9. KV cache scatter kernel (one per layer with same dimensions)
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&scatter_key) {
            let kernel_type = KernelType::KvCacheScatter {
                num_kv_heads,
                head_dim,
                max_len,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(scatter_key, module);
        }

        // 10. Incremental attention kernel (direct mode - for non-graph path)
        let attn_key = format!(
            "incremental_attention_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );
        if !self.modules.contains_key(&attn_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(attn_key, module);
        }

        // CORRECTNESS-010: Preload indirect attention kernel for CUDA graph capture
        // The indirect version reads seq_len from device memory (position_buf)
        let attn_indirect_key = format!(
            "incremental_attention_indirect_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );
        if !self.modules.contains_key(&attn_indirect_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(attn_indirect_key, module);
        }

        // CORRECTNESS-009: Preload multi-warp attention kernels for head_dim > 64
        // Multi-warp kernel handles 4 elements per thread (vs 2 for single-warp)
        // Required for models like Qwen 2.5 with head_dim=128
        let num_warps_per_head = 4u32;
        let multi_warp_key = format!(
            "multi_warp_attention_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
        );
        if !self.modules.contains_key(&multi_warp_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                num_warps_per_head,
                indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(multi_warp_key, module);
        }

        // CORRECTNESS-010: Preload multi-warp indirect attention for graph capture
        let multi_warp_indirect_key = format!(
            "multi_warp_attention_indirect_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
        );
        if !self.modules.contains_key(&multi_warp_indirect_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                num_warps_per_head,
                indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(multi_warp_indirect_key, module);
        }

        if verbose() {
            eprintln!(
                "[PAR-054-FIX] Pre-loaded {} kernel modules for {} layers",
                self.modules.len(),
                num_layers
            );
        }
        Ok(())
    }

    /// PAR-054: Try to capture CUDA graph
    fn try_graph_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Begin graph capture
        self.stream.begin_capture(CaptureMode::Global)?;

        // Run workspace forward pass (all kernels will be captured)
        let capture_result = self.forward_workspace_captured(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        // End capture regardless of result
        let graph = self.stream.end_capture()?;

        // Check capture result
        capture_result?;

        // Instantiate the graph
        let graph_exec = graph.instantiate()?;
        self.decode_graph = Some(graph_exec);
        self.decode_token_count = 1;

        if verbose() {
            eprintln!(
                "[PAR-054] ✓ CUDA graph captured successfully ({} layers + LM head)",
                num_layers
            );
        }

        Ok(())
    }

    /// PAR-054: Replay captured graph with updated position
    fn forward_graphed_replay(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Stateless GPU mode - force position=0, seq_len=1
        static STATELESS_MODE_REPLAY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_stateless = *STATELESS_MODE_REPLAY.get_or_init(|| {
            std::env::var("STATELESS_GPU")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // Update position buffer (async memcpy, doesn't invalidate graph)
        // CORRECTNESS-013: In stateless mode, always use position=0
        if let Some(ref mut pos_buf) = self.position_buf {
            let pos_to_write = if use_stateless { 0 } else { position };
            pos_buf.copy_from_host(&[pos_to_write])?;
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        // The attention kernel reads seq_len from device memory in indirect mode
        // CORRECTNESS-013: In stateless mode, always use seq_len=1
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            let seq_len = if use_stateless { 1 } else { position + 1 };
            seq_len_buf.copy_from_host(&[seq_len])?;
        }

        // Update input buffer
        if let Some(ref mut input_buf) = self.graph_input_buf {
            input_buf.copy_from_host(input)?;
        }

        // Launch captured graph
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }

        self.decode_token_count += 1;

        // Sync and download
        self.stream.synchronize()?;
        if let Some(ref logits_buf) = self.workspace.logits_buf {
            logits_buf.copy_to_host(logits)?;
        }

        Ok(())
    }

    /// PAR-062: GPU-side argmax to eliminate logits transfer bottleneck
    ///
    /// Instead of copying all 152064 logits (600KB) from GPU to CPU for argmax,
    /// this method runs argmax entirely on GPU and only copies the result token ID (4 bytes).
    /// This is a 150,000x reduction in data transfer per token.
    ///
    /// # Algorithm
    ///
    /// Two-pass reduction:
    /// 1. Block-level: Each block finds local (max_val, max_idx) using shared memory
    /// 2. Final: Single block reduces block results to find global argmax
    ///
    /// # Arguments
    ///
    /// * `logits_ptr` - Device pointer to logits (vocab_size f32s)
    /// * `vocab_size` - Number of vocabulary entries (e.g., 152064)
    ///
    /// # Returns
    ///
    /// The token ID with the maximum logit value
    pub fn gpu_argmax(&mut self, logits_ptr: u64, vocab_size: u32) -> Result<u32, GpuError> {
        // PAR-068: Optimized GPU argmax with pre-allocated buffers
        // Eliminates 3 GPU allocations per token and removes intermediate sync
        let block_size = 256u32;
        let elements_per_block = block_size * 4; // 4 elements per thread
        let num_blocks = (vocab_size + elements_per_block - 1) / elements_per_block;

        // PAR-068: Lazy allocate argmax buffers on first use, reuse thereafter
        if self.argmax_block_vals.is_none() || self.argmax_num_blocks != num_blocks {
            self.argmax_block_vals = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_block_idxs = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_result = Some(GpuBuffer::new(&self.context, 1)?);
            self.argmax_num_blocks = num_blocks;
        }

        let block_max_vals = self
            .argmax_block_vals
            .as_ref()
            .expect("argmax_block_vals must be initialized");
        let block_max_idxs = self
            .argmax_block_idxs
            .as_ref()
            .expect("argmax_block_idxs must be initialized");
        let result_buf = self
            .argmax_result
            .as_ref()
            .expect("argmax_result must be initialized");

        // Load first-pass kernel module (cached after first use)
        let argmax_kernel_type = KernelType::ArgMax { length: vocab_size };
        let argmax_key = format!("argmax_{}", vocab_size);
        if !self.modules.contains_key(&argmax_key) {
            let ptx = self.kernels.generate_ptx(&argmax_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(argmax_key.clone(), module);
        }

        // Load second-pass kernel module (cached after first use)
        let final_kernel_type = KernelType::ArgMaxFinal { num_blocks };
        let final_key = format!("argmax_final_{}", num_blocks);
        if !self.modules.contains_key(&final_key) {
            let ptx = self.kernels.generate_ptx(&final_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(final_key.clone(), module);
        }

        // Prepare kernel arguments
        let kernel_name = self.kernels.kernel_name(&argmax_kernel_type);
        // PAR-068-FIX: Do NOT use .with_shared_mem() - PTX declares static shared memory via .shared directive
        let launch_config = LaunchConfig::grid_2d(num_blocks, 1, block_size, 1);

        let mut input_ptr = logits_ptr;
        let mut block_vals_ptr = block_max_vals.as_ptr();
        let mut block_idxs_ptr = block_max_idxs.as_ptr();
        let mut length_val = vocab_size;

        // Launch first-pass kernel (block-level reduction)
        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let module = self
                .modules
                .get_mut(&argmax_key)
                .expect("argmax module just inserted");
            self.stream.launch_kernel(
                module,
                kernel_name,
                &launch_config,
                &mut [
                    std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: NO intermediate sync - launch both kernels back-to-back
        // The kernels are on the same stream, so execution is serialized

        // Launch second-pass kernel (final reduction)
        let final_kernel_name = self.kernels.kernel_name(&final_kernel_type);
        let final_launch_config = LaunchConfig::grid_2d(1, 1, 256, 1);

        let mut result_ptr = result_buf.as_ptr();
        let mut num_blocks_val = num_blocks;

        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let final_module = self
                .modules
                .get_mut(&final_key)
                .expect("argmax_final module just inserted");
            self.stream.launch_kernel(
                final_module,
                final_kernel_name,
                &final_launch_config,
                &mut [
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut result_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut num_blocks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: Single sync after both kernels complete
        self.stream.synchronize()?;
        let mut result = [0u32];
        result_buf.copy_to_host(&mut result)?;

        // CORRECTNESS-005: Debug GPU argmax result
        static ARGMAX_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ARGMAX_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            eprintln!(
                "[CORRECTNESS-005] GPU argmax: token_id={}, vocab_size={}",
                result[0], vocab_size
            );
        }

        Ok(result[0])
    }

    /// PAR-062: Forward pass with GPU-side argmax returning token ID directly
    ///
    /// Like `forward_graphed_replay` but uses GPU argmax instead of downloading all logits.
    /// Reduces data transfer from 600KB to 4 bytes per token.
    ///
    /// # Performance Target
    ///
    /// - Before: ~3ms logits transfer per token on PCIe
    /// - After: ~0.001ms token ID transfer
    /// - Expected speedup: ~1.2x overall throughput
    pub fn forward_graphed_replay_to_token_id(
        &mut self,
        input: &[f32],
        position: u32,
        vocab_size: u32,
    ) -> Result<u32, GpuError> {
        // PAR-068: Use GPU argmax to eliminate 600KB D2H transfer bottleneck
        // Root cause fix: Removed .with_shared_mem() from argmax kernel launch configs
        // (PTX declares static shared memory, mixing with dynamic causes CUDA_ERROR_UNKNOWN)

        // PAR-072: Use ASYNC H2D copies to eliminate blocking overhead
        // Root cause: cuMemcpyHtoD has ~10-30µs overhead per call
        // Fix: Use cuMemcpyHtoDAsync on the same stream as graph launch

        // Update position buffer (async memcpy on same stream)
        if let Some(ref mut pos_buf) = self.position_buf {
            // SAFETY: position is stack-allocated and we synchronize before returning
            unsafe {
                pos_buf.copy_from_host_async(&[position], &self.stream)?;
            }
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        let seq_len = position + 1;
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            // SAFETY: seq_len is stack-allocated and we synchronize before returning
            unsafe {
                seq_len_buf.copy_from_host_async(&[seq_len], &self.stream)?;
            }
        }

        // Update input buffer (async - largest copy, ~14KB for Qwen 0.5B)
        if let Some(ref mut input_buf) = self.graph_input_buf {
            // SAFETY: input slice is valid for the duration of this function
            // and we synchronize in gpu_argmax before returning
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                input_buf.copy_from_host_async(input, &self.stream)?;
            }
        }

        // Launch captured graph (all H2D copies are ordered before this on same stream)
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }

        self.decode_token_count += 1;

        // PAR-068: GPU argmax instead of downloading 600KB logits
        // This reduces D2H transfer from 600KB to 4 bytes per token
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidParameter("logits_buf not allocated".into()))?
            .as_ptr();

        // CORRECTNESS-004: Debug graph-replayed logits and compare argmax
        static GPU_DEBUG_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_enabled = *GPU_DEBUG_FLAG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if debug_enabled {
            self.stream.synchronize()?;
            // Download ALL logits to compute CPU argmax for comparison
            let mut all_logits = vec![0.0f32; vocab_size as usize];
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let debug_view =
                unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, vocab_size as usize) };
            debug_view.copy_to_host(&mut all_logits)?;
            std::mem::forget(debug_view);

            // CPU argmax
            let (cpu_argmax_idx, cpu_argmax_val) = all_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("CUDA operation failed"))
                .expect("CUDA operation failed");

            eprintln!(
                "[CORRECTNESS-004] Graph logits[0..20]: {:?}",
                &all_logits[..20]
            );
            eprintln!(
                "[CORRECTNESS-004] GPU argmax: idx={}, val={}",
                cpu_argmax_idx, cpu_argmax_val
            );

            // Compare against CPU's expected top tokens: 19 ("4"), 17 ("2"), 785 (" The")
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 19 ('4'): {}",
                all_logits.get(19).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 17 ('2'): {}",
                all_logits.get(17).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 785: {}",
                all_logits.get(785).unwrap_or(&f32::NAN)
            );

            // Top 5 GPU logits
            let mut top5: Vec<(usize, f32)> = all_logits.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top5.truncate(10);
            eprintln!("[CORRECTNESS-004] GPU top10 logits: {:?}", top5);
        }

        let gpu_result = self.gpu_argmax(logits_ptr, vocab_size)?;

        if debug_enabled {
            eprintln!("[CORRECTNESS-004] GPU argmax result: {}", gpu_result);
        }

        Ok(gpu_result)
    }

    /// PAR-054: Forward pass for graph capture (uses pre-allocated workspace)
    ///
    /// # Safety
    ///
    /// This function must only be called while stream capture is active.
    /// All output buffers (workspace.logits_buf) must be pre-allocated before capture.
    fn forward_workspace_captured(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Layer 0: input from graph_input_buf
        // PAR-070: Position is read from position_buf in indirect mode (graph capture)
        // The position parameter here is ignored since position_buf.is_some() triggers indirect mode
        if num_layers > 0 {
            let input_ptr = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .as_ptr();
            let input_len = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };
            let layer_weights = self.indexed_layer_weights[0].clone();
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                0,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Layers 1+: input from hidden_buf2
        for layer_idx in 1..num_layers {
            let layer_weights = self.indexed_layer_weights[layer_idx].clone();
            let buf_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let buf_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                layer_idx,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Output RMSNorm - PAR-054: Use pre-allocated normed_hidden_buf
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-054: output_norm not cached".to_string())
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        let hidden_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .as_ptr();
        let hidden_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_input = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };

        // PAR-054: Write to pre-allocated normed_hidden_buf (no allocation during capture)
        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_output = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        self.rmsnorm_ptr_into(
            &hidden_input,
            output_gamma_ptr,
            output_gamma_len,
            &normed_output,
            hidden_dim,
            epsilon,
        )?;
        std::mem::forget(hidden_input);
        std::mem::forget(normed_output);

        // LM head projection - PAR-054: Use pre-allocated logits_buf
        // PAR-058: Use correct kernel based on LM head quantization type
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .as_ptr();
        let logits_len = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let logits_output = unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, logits_len) };

        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_input = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        // PAR-058: Dispatch to correct kernel based on LM head quant type
        // Validate qtype against actual size - GGUF metadata can lie!
        let lm_head_qtype =
            WeightQuantType::from_size(self.lm_head_len, vocab_size as usize, hidden_dim as usize)
                .unwrap_or(self.lm_head_qtype);

        // Log if we overrode the type
        if lm_head_qtype != self.lm_head_qtype {
            eprintln!(
                "[PAR-058] LM head qtype override: {:?} -> {:?} (size-based detection)",
                self.lm_head_qtype, lm_head_qtype
            );
        }

        match lm_head_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
        }

        // PAR-064-FIX: Add LM head bias after GEMV (if present)
        // Without this, GPU inference produces incorrect token predictions
        if self.lm_head_bias_ptr != 0 && self.lm_head_bias_len > 0 {
            // Create non-owning buffer wrapper from device pointer
            // SAFETY: bias_ptr is valid device memory owned by bias_cache
            let bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(self.lm_head_bias_ptr, self.lm_head_bias_len)
            };

            // Add bias in-place: logits = logits + bias
            self.residual_add_into(&logits_output, &bias_buf, &logits_output, vocab_size)?;

            // Prevent Drop from freeing borrowed memory
            std::mem::forget(bias_buf);
        }

        std::mem::forget(normed_input);
        std::mem::forget(logits_output);

        Ok(())
    }
}
