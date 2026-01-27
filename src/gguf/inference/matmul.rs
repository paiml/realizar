//! Quantized matrix operations for OwnedQuantizedModel
//!
//! Contains embed, fused_matmul, qkv_matmul methods with real implementations
//! for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K quantization formats.

use crate::error::{RealizarError, Result};
use crate::gguf::types::{
    GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_0,
};
use crate::gguf::{ops, OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedTensor};

impl OwnedQuantizedModel {
    /// Look up token embeddings (public for debugging PAR-001)
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// Look up single token embedding into pre-allocated buffer (IMP-131)
    pub(crate) fn embed_into(&self, token_id: u32, output: &mut [f32]) {
        let hidden_dim = self.config.hidden_dim;
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        if end <= self.token_embedding.len() {
            output[..hidden_dim].copy_from_slice(&self.token_embedding[start..end]);
        } else {
            output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Fused dequantize + matmul for quantized weights
    ///
    /// Supports Q4_0, Q8_0, Q4_1, Q5_0, Q4_K, Q5_K, Q6_K quantization formats.
    /// Uses SIMD-accelerated implementations for optimal performance.
    pub(crate) fn fused_matmul(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        use crate::quantize::{
            dequantize_q4_1, dequantize_q5_0, fused_q4_0_q8_0_parallel_matvec,
            fused_q4k_parallel_matvec, fused_q5k_parallel_matvec, fused_q6k_parallel_matvec,
            fused_q8_0_q8_0_parallel_matvec,
        };
        use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // CUDA path when enabled
        #[cfg(feature = "cuda")]
        if let Some(ref executor_mutex) = self.cuda_executor {
            return self.fused_matmul_cuda(input, weight, executor_mutex);
        }

        // CPU path: For Q4_0, use fused Q8_0 integer SIMD matmul (llama.cpp parity)
        if weight.qtype == GGUF_TYPE_Q4_0 {
            if seq_len == 1 {
                return fused_q4_0_q8_0_parallel_matvec(&weight.data, input, in_dim, out_dim);
            }
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = fused_q4_0_q8_0_parallel_matvec(&weight.data, x, in_dim, out_dim)?;
                output.extend_from_slice(&row_output);
            }
            return Ok(output);
        }

        // CPU path: For Q8_0, use fused Q8_0 × Q8_0 integer SIMD matmul
        if weight.qtype == GGUF_TYPE_Q8_0 {
            if seq_len == 1 {
                return fused_q8_0_q8_0_parallel_matvec(&weight.data, input, in_dim, out_dim);
            }
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = fused_q8_0_q8_0_parallel_matvec(&weight.data, x, in_dim, out_dim)?;
                output.extend_from_slice(&row_output);
            }
            return Ok(output);
        }

        // CPU path: For Q4_1, use dequantize + SIMD matmul
        if weight.qtype == GGUF_TYPE_Q4_1 {
            let weights_f32 = dequantize_q4_1(&weight.data)?;

            let weight_matrix = match TruenoMatrix::from_vec(out_dim, in_dim, weights_f32) {
                Ok(m) => m,
                Err(_) => {
                    return Err(RealizarError::InvalidShape {
                        reason: "Failed to create weight matrix for Q4_1".to_string(),
                    });
                },
            };

            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let x_vec = TruenoVector::from_slice(x);
                match weight_matrix.matvec(&x_vec) {
                    Ok(r) => output.extend_from_slice(r.as_slice()),
                    Err(_) => {
                        return Err(RealizarError::InvalidShape {
                            reason: "SIMD matvec failed for Q4_1".to_string(),
                        });
                    },
                }
            }
            return Ok(output);
        }

        // CPU path: For Q5_0, use dequantize + SIMD matmul
        if weight.qtype == GGUF_TYPE_Q5_0 {
            let weights_f32 = dequantize_q5_0(&weight.data)?;

            let weight_matrix = match TruenoMatrix::from_vec(out_dim, in_dim, weights_f32) {
                Ok(m) => m,
                Err(_) => {
                    return Err(RealizarError::InvalidShape {
                        reason: "Failed to create weight matrix for Q5_0".to_string(),
                    });
                },
            };

            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let x_vec = TruenoVector::from_slice(x);
                match weight_matrix.matvec(&x_vec) {
                    Ok(r) => output.extend_from_slice(r.as_slice()),
                    Err(_) => {
                        return Err(RealizarError::InvalidShape {
                            reason: "SIMD matvec failed for Q5_0".to_string(),
                        });
                    },
                }
            }
            return Ok(output);
        }

        // CPU path: Process each position in sequence for Q4_K, Q5_K, Q6_K
        if seq_len > 1 {
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = match weight.qtype {
                    GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    _ => {
                        return Err(RealizarError::UnsupportedOperation {
                            operation: "owned_fused_matmul".to_string(),
                            reason: format!(
                                "Fused matmul only supports Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                                weight.qtype
                            ),
                        });
                    },
                };
                output.extend_from_slice(&row_output);
            }
            Ok(output)
        } else {
            // Single position - most common case in generation
            match weight.qtype {
                GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                _ => Err(RealizarError::UnsupportedOperation {
                    operation: "owned_fused_matmul".to_string(),
                    reason: format!(
                        "Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                        weight.qtype
                    ),
                }),
            }
        }
    }

    /// CUDA path for fused matmul
    #[cfg(feature = "cuda")]
    fn fused_matmul_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        executor_mutex: &std::sync::Mutex<crate::cuda::CudaExecutor>,
    ) -> Result<Vec<f32>> {
        use tracing::info_span;

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;
        let gemm_start = std::time::Instant::now();
        let mut output = vec![0.0f32; seq_len * out_dim];

        // Use native quantized GEMV kernels for single-token generation
        if seq_len == 1 {
            let cache_key = format!(
                "{}_{:016x}",
                match weight.qtype {
                    GGUF_TYPE_Q4_K => "q4k",
                    GGUF_TYPE_Q5_K => "q5k",
                    GGUF_TYPE_Q6_K => "q6k",
                    _ => "unknown",
                },
                weight.data.as_ptr() as usize
            );

            if weight.qtype == GGUF_TYPE_Q4_K
                || weight.qtype == GGUF_TYPE_Q5_K
                || weight.qtype == GGUF_TYPE_Q6_K
            {
                let mut executor =
                    executor_mutex
                        .lock()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_lock".to_string(),
                            reason: format!("Failed to acquire CUDA executor lock: {e}"),
                        })?;

                executor
                    .make_current()
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_make_current".to_string(),
                        reason: format!("Failed to set CUDA context current: {e}"),
                    })?;

                if !executor.has_quantized_weights(&cache_key) {
                    executor
                        .load_quantized_weights(&cache_key, &weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_cache".to_string(),
                            reason: format!("Failed to cache weights: {e}"),
                        })?;
                }

                let result = match weight.qtype {
                    GGUF_TYPE_Q4_K => executor.q4k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    GGUF_TYPE_Q5_K => executor.q5k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    GGUF_TYPE_Q6_K => executor.q6k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    _ => unreachable!(),
                };

                result.map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_gemv".to_string(),
                    reason: format!("CUDA GEMV failed: {e}"),
                })?;

                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;
                let _span = info_span!(
                    "gpu_kernel:gemv",
                    gpu.backend = "cuda",
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }
        }

        // Fallback: Dequantize and use FP32 GEMM
        let dequant_weight = self.dequantize_weight_for_cuda(weight)?;

        {
            let mut executor =
                executor_mutex
                    .lock()
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_gemm_lock".to_string(),
                        reason: format!("Failed to acquire CUDA executor lock: {e}"),
                    })?;

            executor
                .make_current()
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_make_current".to_string(),
                    reason: format!("Failed to set CUDA context current: {e}"),
                })?;

            executor
                .gemm(
                    input,
                    &dequant_weight,
                    &mut output,
                    seq_len as u32,
                    out_dim as u32,
                    in_dim as u32,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_gemm".to_string(),
                    reason: format!("CUDA GEMM failed: {e}"),
                })?;
        }

        self.cuda_kernel_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(output)
    }

    /// Fused matmul into pre-allocated output buffer
    pub(crate) fn fused_matmul_into(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        output: &mut [f32],
    ) -> Result<()> {
        use crate::quantize::{
            fused_q4_0_q8_0_parallel_matvec_into, fused_q4k_parallel_matvec_into,
            fused_q5k_parallel_matvec_into, fused_q6k_parallel_matvec_into,
            fused_q8_0_q8_0_parallel_matvec_into,
        };

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // Only support single-token case for now (most common in generation)
        if seq_len != 1 {
            let result = self.fused_matmul(input, weight)?;
            output[..result.len()].copy_from_slice(&result);
            return Ok(());
        }

        debug_assert!(
            output.len() >= out_dim,
            "Output buffer too small: {} < {}",
            output.len(),
            out_dim
        );

        match weight.qtype {
            GGUF_TYPE_Q4_0 => fused_q4_0_q8_0_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q8_0 => fused_q8_0_q8_0_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            _ => {
                let result = self.fused_matmul(input, weight)?;
                output[..result.len()].copy_from_slice(&result);
                Ok(())
            },
        }
    }

    /// QKV projection matmul
    pub fn qkv_matmul(&self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let seq_len = input.len() / hidden_dim;

                let q_out = self.fused_matmul(input, q)?;
                let k_out = self.fused_matmul(input, k)?;
                let v_out = self.fused_matmul(input, v)?;

                // Interleave Q, K, V for each position
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(seq_len * qkv_dim);
                for s in 0..seq_len {
                    output.extend_from_slice(&q_out[s * q.out_dim..(s + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[s * k.out_dim..(s + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[s * v.out_dim..(s + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// QKV matmul into pre-allocated buffer
    pub fn qkv_matmul_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
    ) -> Result<()> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_into(input, weight, output),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.out_dim;
                let k_dim = k.out_dim;
                let v_dim = v.out_dim;

                self.fused_matmul_into(input, q, &mut output[..q_dim])?;
                self.fused_matmul_into(input, k, &mut output[q_dim..q_dim + k_dim])?;
                self.fused_matmul_into(
                    input,
                    v,
                    &mut output[q_dim + k_dim..q_dim + k_dim + v_dim],
                )?;

                Ok(())
            },
        }
    }

    /// Layer normalization
    pub fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        ops::layer_norm(input, weight, bias, eps)
    }

    /// Add bias to activations
    pub fn add_bias(&self, input: &mut [f32], bias: &[f32]) {
        for (x, b) in input.iter_mut().zip(bias.iter()) {
            *x += b;
        }
    }

    /// GELU activation
    pub fn gelu(&self, input: &mut [f32]) {
        for x in input.iter_mut() {
            *x = 0.5 * *x * (1.0 + (*x * 0.797_884_6 * (1.0 + 0.044715 * *x * *x)).tanh());
        }
    }

    /// Fused RMSNorm + matmul helper
    fn fused_rmsnorm_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights (most common)
        if weight.qtype == GGUF_TYPE_Q4_0 && input.len() == weight.in_dim {
            return fused_rmsnorm_q4_0_matmul(
                input,
                norm_weight,
                eps,
                &weight.data,
                weight.in_dim,
                weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, norm_weight, eps);
        self.fused_matmul(&normed, weight)
    }

    /// Fused RMSNorm + QKV matmul
    pub fn fused_rmsnorm_qkv_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        qkv: &OwnedQKVWeights,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_rmsnorm_matmul(input, norm_weight, eps, weight)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, normalize once and reuse
                let normed = ops::rms_norm(input, norm_weight, eps);

                // PMAT-114: Trace K weight to compare with APR
                static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if std::env::var("APR_TRACE_WEIGHTS").is_ok() && !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[PMAT-114-GGUF] K weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                        k.in_dim, k.out_dim, k.qtype, k.data.len());
                    // Dequantize first row completely to compare with APR
                    // Q4K: 144 bytes per super-block of 256 values, so first row = in_dim/256 super-blocks
                    let bytes_per_row = (k.in_dim.div_ceil(256)) * 144;
                    use crate::quantize::dequantize_q4_k_parallel;
                    if let Ok(dequant) = dequantize_q4_k_parallel(&k.data[0..bytes_per_row]) {
                        let row_mean: f32 = dequant.iter().sum::<f32>() / dequant.len() as f32;
                        let row_min = dequant.iter().cloned().fold(f32::INFINITY, f32::min);
                        let row_max = dequant.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!("[PMAT-114-GGUF] K weight row 0 (dequant): mean={:.6}, min={:.6}, max={:.6}, len={}",
                            row_mean, row_min, row_max, dequant.len());
                        eprintln!("[PMAT-114-GGUF] K weight row 0 first10={:?}", &dequant[..10.min(dequant.len())]);
                    }
                }

                let q_out = self.fused_matmul(&normed, q)?;
                let k_out = self.fused_matmul(&normed, k)?;
                let v_out = self.fused_matmul(&normed, v)?;

                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(qkv_dim);
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// Fused RMSNorm + LM head
    pub fn fused_rmsnorm_lm_head(&self, input: &[f32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights
        if self.lm_head_weight.qtype == GGUF_TYPE_Q4_0 && input.len() == self.lm_head_weight.in_dim
        {
            return fused_rmsnorm_q4_0_matmul(
                input,
                &self.output_norm_weight,
                self.config.eps,
                &self.lm_head_weight.data,
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, &self.output_norm_weight, self.config.eps);
        self.fused_matmul(&normed, &self.lm_head_weight)
    }

    /// Fused RMSNorm + FFN up/gate projections for SwiGLU
    pub fn fused_rmsnorm_ffn_up_gate(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        up_weight: &OwnedQuantizedTensor,
        gate_weight: &OwnedQuantizedTensor,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::quantize::fused_rmsnorm_ffn_up_gate;

        // Only use fused path for Q4_0 weights
        if up_weight.qtype == GGUF_TYPE_Q4_0
            && gate_weight.qtype == GGUF_TYPE_Q4_0
            && input.len() == up_weight.in_dim
            && up_weight.in_dim == gate_weight.in_dim
            && up_weight.out_dim == gate_weight.out_dim
        {
            return fused_rmsnorm_ffn_up_gate(
                input,
                norm_weight,
                eps,
                &up_weight.data,
                &gate_weight.data,
                up_weight.in_dim,
                up_weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmuls for other types
        let normed = ops::rms_norm(input, norm_weight, eps);
        let up_out = self.fused_matmul(&normed, up_weight)?;
        let gate_out = self.fused_matmul(&normed, gate_weight)?;
        Ok((up_out, gate_out))
    }

    /// Q8K QKV matmul into buffer
    ///
    /// Uses pre-quantized Q8K activations for faster matmul with Q4K weights.
    #[allow(unused_variables)]
    pub fn qkv_matmul_q8k_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
        scales: &[f32],
        quants: &[i8],
    ) -> Result<()> {
        // Fall back to regular qkv_matmul_into (Q8K acceleration deferred)
        self.qkv_matmul_into(input, qkv, output)
    }

    /// Helper to dequantize weights for CUDA GEMM
    #[cfg(feature = "cuda")]
    fn dequantize_weight_for_cuda(&self, weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        use crate::quantize::{
            dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_k,
            dequantize_q6_k, dequantize_q8_0,
        };

        match weight.qtype {
            GGUF_TYPE_Q4_0 => dequantize_q4_0(&weight.data),
            GGUF_TYPE_Q4_1 => dequantize_q4_1(&weight.data),
            GGUF_TYPE_Q5_0 => dequantize_q5_0(&weight.data),
            GGUF_TYPE_Q8_0 => dequantize_q8_0(&weight.data),
            GGUF_TYPE_Q4_K => dequantize_q4_k(&weight.data),
            GGUF_TYPE_Q5_K => dequantize_q5_k(&weight.data),
            GGUF_TYPE_Q6_K => dequantize_q6_k(&weight.data),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "dequantize_weight_for_cuda".to_string(),
                reason: format!("Unsupported quantization type: {}", weight.qtype),
            }),
        }
    }
}
