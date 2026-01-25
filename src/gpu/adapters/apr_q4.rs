//! Q4_0 GPU Adapter - Direct Quantized Inference (PMAT-803)
//!
//! Keeps weights in Q4_0 format on GPU, using `q4_0_gemv_into` kernels.
//! This eliminates the 8x bandwidth overhead of F32 dequantization.
//!
//! # Performance Target
//!
//! >50 tok/s on TinyLlama-1.1B (4x improvement over F32 GPU)
//!
//! # Architecture
//!
//! ```text
//! QuantizedAprTransformerQ4 (CPU, Q4_0 weights)
//!         │
//!         ▼ AprQ4ToGpuAdapter::upload_weights()
//!         │
//! CudaExecutor (GPU, Q4_0 cached)
//!         │
//!         ▼ GpuModelQ4::forward()
//!         │
//! Logits (F32)
//! ```

#![allow(clippy::similar_names)]

use crate::apr_transformer::{AprTransformerConfig, QuantizedAprTransformerQ4};
use crate::error::{RealizarError, Result};
#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Q4_0 quantization type ID (GGML format)
#[cfg(feature = "cuda")]
const Q4_0_TYPE: u32 = 2;

/// Adapter for uploading Q4_0 APR models to GPU
///
/// Unlike `AprToGpuAdapter`, this keeps weights quantized on GPU.
pub struct AprQ4ToGpuAdapter;

impl AprQ4ToGpuAdapter {
    /// Upload Q4_0 weights to CUDA executor
    ///
    /// Weights are cached on GPU by layer name for reuse across forward passes.
    /// No dequantization is performed - weights stay in Q4_0 format.
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with Q4_0 weights
    /// * `executor` - CUDA executor with weight cache
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    ///
    /// # Errors
    ///
    /// Returns error if GPU upload fails.
    #[cfg(feature = "cuda")]
    pub fn upload_weights(
        apr: &QuantizedAprTransformerQ4,
        executor: &mut CudaExecutor,
    ) -> Result<usize> {
        let mut total_bytes = 0;

        // Upload layer weights (Q4_0 only - norms are kept on CPU)
        for (layer_idx, layer) in apr.layers.iter().enumerate() {
            // QKV projection
            let qkv_name = format!("layer_{layer_idx}.attn.qkv");
            let qkv_bytes = executor
                .load_quantized_weights_with_type(&qkv_name, &layer.qkv_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload {qkv_name}: {e}") })?;
            total_bytes += qkv_bytes;

            // Output projection
            let out_name = format!("layer_{layer_idx}.attn.out");
            let out_bytes = executor
                .load_quantized_weights_with_type(&out_name, &layer.attn_output_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload {out_name}: {e}") })?;
            total_bytes += out_bytes;

            // FFN up projection
            let up_name = format!("layer_{layer_idx}.ffn.up");
            let up_bytes = executor
                .load_quantized_weights_with_type(&up_name, &layer.ffn_up_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload {up_name}: {e}") })?;
            total_bytes += up_bytes;

            // FFN down projection
            let down_name = format!("layer_{layer_idx}.ffn.down");
            let down_bytes = executor
                .load_quantized_weights_with_type(&down_name, &layer.ffn_down_weight.data, Q4_0_TYPE)
                .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload {down_name}: {e}") })?;
            total_bytes += down_bytes;

            // FFN gate projection (optional, for SwiGLU)
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let gate_name = format!("layer_{layer_idx}.ffn.gate");
                let gate_bytes = executor
                    .load_quantized_weights_with_type(&gate_name, &gate_weight.data, Q4_0_TYPE)
                    .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload {gate_name}: {e}") })?;
                total_bytes += gate_bytes;
            }
        }

        // Upload LM head (Q4_0)
        let lm_head_bytes = executor
            .load_quantized_weights_with_type("lm_head", &apr.lm_head_weight.data, Q4_0_TYPE)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload lm_head: {e}") })?;
        total_bytes += lm_head_bytes;

        Ok(total_bytes)
    }

    /// Create a GPU model wrapper for Q4_0 inference
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer (for config, embeddings, and norm weights)
    ///
    /// # Returns
    ///
    /// `GpuModelQ4` ready for inference
    #[must_use]
    pub fn create_model(apr: &QuantizedAprTransformerQ4) -> GpuModelQ4 {
        // Extract layer norms (small, keep on CPU)
        let layer_norms: Vec<LayerNorms> = apr.layers.iter().map(|layer| {
            LayerNorms {
                attn_norm: layer.attn_norm_weight.clone(),
                ffn_norm: layer.ffn_norm_weight.clone().unwrap_or_else(|| vec![1.0; apr.config.hidden_dim]),
            }
        }).collect();

        GpuModelQ4 {
            config: apr.config.clone(),
            token_embedding: apr.token_embedding.clone(),
            output_norm_weight: apr.output_norm_weight.clone(),
            layer_norms,
            num_layers: apr.layers.len(),
            has_gate: apr.layers.first().is_some_and(|l| l.ffn_gate_weight.is_some()),
        }
    }
}

/// Layer normalization weights (small, kept on CPU)
#[derive(Debug, Clone)]
pub struct LayerNorms {
    /// Attention norm weight
    pub attn_norm: Vec<f32>,
    /// FFN norm weight
    pub ffn_norm: Vec<f32>,
}

/// GPU model for Q4_0 inference
///
/// Lightweight wrapper that holds config and uses CudaExecutor for computation.
/// Large Q4_0 weights are stored on GPU, small norm weights on CPU.
#[derive(Debug, Clone)]
pub struct GpuModelQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32, CPU copy for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Output norm weight (F32, CPU)
    pub output_norm_weight: Vec<f32>,
    /// Per-layer norm weights (F32, CPU)
    pub layer_norms: Vec<LayerNorms>,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Whether model uses SwiGLU (has gate projection)
    pub has_gate: bool,
}

/// Helper to download GPU buffer to host Vec
#[cfg(feature = "cuda")]
fn gpu_to_host(buf: &GpuBuffer<f32>) -> Result<Vec<f32>> {
    let mut host = vec![0.0f32; buf.len()];
    buf.copy_to_host(&mut host)
        .map_err(|e| RealizarError::GpuError { reason: format!("GPU->CPU copy failed: {e}") })?;
    Ok(host)
}

impl GpuModelQ4 {
    /// Execute forward pass using Q4_0 kernels
    ///
    /// # Arguments
    ///
    /// * `executor` - CUDA executor with cached weights
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction
    #[cfg(feature = "cuda")]
    pub fn forward(
        &self,
        executor: &mut CudaExecutor,
        token_ids: &[usize],
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // 1. Embed tokens (CPU - fast lookup)
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let start = token_id * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                // Out of vocab, use zeros
                hidden.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        // 2. Upload hidden state to GPU
        let mut hidden_gpu = GpuBuffer::from_host(executor.context(), &hidden)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload hidden: {e}") })?;

        // 3. Pass through transformer layers
        for layer_idx in 0..self.num_layers {
            hidden_gpu = self.forward_layer(executor, &hidden_gpu, layer_idx, seq_len)?;
        }

        // 4. Final layer norm (CPU for simplicity - small operation)
        let mut hidden = gpu_to_host(&hidden_gpu)?;

        // RMSNorm
        self.rms_norm_inplace(&mut hidden, &self.output_norm_weight);

        // 5. LM head projection (GPU, Q4_0)
        let hidden_gpu = GpuBuffer::from_host(executor.context(), &hidden)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload normed hidden: {e}") })?;

        let logits_gpu = GpuBuffer::new(executor.context(), vocab_size)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate logits: {e}") })?;

        let lm_head_ptr = executor.get_quantized_weight_ptr("lm_head")
            .map_err(|e| RealizarError::GpuError { reason: format!("lm_head not cached: {e}") })?;

        executor.q4_0_gemv_into(
            lm_head_ptr,
            &hidden_gpu,
            &logits_gpu,
            vocab_size as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("LM head GEMV failed: {e}") })?;

        // 6. Sync and return
        executor.synchronize()
            .map_err(|e| RealizarError::GpuError { reason: format!("Sync failed: {e}") })?;

        gpu_to_host(&logits_gpu)
    }

    /// Execute single transformer layer
    #[cfg(feature = "cuda")]
    fn forward_layer(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<GpuBuffer<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;

        // 1. Download for attention norm (CPU)
        let hidden = gpu_to_host(input)?;

        // RMSNorm using cached layer norms
        let mut normed = hidden.clone();
        self.rms_norm_inplace(&mut normed, &self.layer_norms[layer_idx].attn_norm);

        // 2. Upload normed input
        let normed_gpu = GpuBuffer::from_host(executor.context(), &normed)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload normed: {e}") })?;

        // 3. QKV projection (Q4_0)
        let qkv_gpu = GpuBuffer::new(executor.context(), qkv_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate QKV: {e}") })?;

        let qkv_name = format!("layer_{layer_idx}.attn.qkv");
        let qkv_ptr = executor.get_quantized_weight_ptr(&qkv_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{qkv_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            qkv_ptr,
            &normed_gpu,
            &qkv_gpu,
            qkv_dim as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("QKV GEMV failed: {e}") })?;

        // 4. Attention (CPU for single-token - GPU launch overhead exceeds benefit)
        let qkv = gpu_to_host(&qkv_gpu)?;

        // Apply RoPE to Q and K before attention
        // For seq_len tokens at layer 0, positions are [0, 1, 2, ... seq_len-1]
        let mut qkv_with_rope = qkv.clone();
        self.apply_rope_to_qkv(&mut qkv_with_rope, seq_len, hidden_dim, num_heads, num_kv_heads);

        let attn_out = self.attention_cpu(&qkv_with_rope, seq_len, hidden_dim, num_heads, num_kv_heads);

        // 5. Output projection (Q4_0)
        let attn_out_gpu = GpuBuffer::from_host(executor.context(), &attn_out)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload attn_out: {e}") })?;

        let out_gpu = GpuBuffer::new(executor.context(), hidden_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate out: {e}") })?;

        let out_name = format!("layer_{layer_idx}.attn.out");
        let out_ptr = executor.get_quantized_weight_ptr(&out_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{out_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            out_ptr,
            &attn_out_gpu,
            &out_gpu,
            hidden_dim as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Out GEMV failed: {e}") })?;

        // 6. Residual connection (CPU)
        let mut out = gpu_to_host(&out_gpu)?;

        for i in 0..hidden_dim {
            out[i] += hidden[i];
        }

        // 7. FFN norm
        let mut ffn_input = out.clone();
        self.rms_norm_inplace(&mut ffn_input, &self.layer_norms[layer_idx].ffn_norm);

        // 8. FFN (GPU, Q4_0)
        let ffn_input_gpu = GpuBuffer::from_host(executor.context(), &ffn_input)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload ffn_input: {e}") })?;

        let ffn_out = if self.has_gate {
            // SwiGLU: gate * up * silu
            self.ffn_swiglu_gpu(executor, &ffn_input_gpu, layer_idx)?
        } else {
            // Standard FFN: up -> activation -> down
            self.ffn_standard_gpu(executor, &ffn_input_gpu, layer_idx)?
        };

        // 9. Final residual
        let mut ffn_out_host = gpu_to_host(&ffn_out)?;

        for i in 0..hidden_dim {
            ffn_out_host[i] += out[i];
        }

        // 10. Upload final output
        GpuBuffer::from_host(executor.context(), &ffn_out_host)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload final output: {e}") })
    }

    /// SwiGLU FFN using Q4_0 kernels
    #[cfg(feature = "cuda")]
    fn ffn_swiglu_gpu(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Gate projection
        let gate_gpu = GpuBuffer::new(executor.context(), intermediate_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate gate: {e}") })?;

        let gate_name = format!("layer_{layer_idx}.ffn.gate");
        let gate_ptr = executor.get_quantized_weight_ptr(&gate_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{gate_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            gate_ptr,
            input,
            &gate_gpu,
            intermediate_dim as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Gate GEMV failed: {e}") })?;

        // Up projection
        let up_gpu = GpuBuffer::new(executor.context(), intermediate_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate up: {e}") })?;

        let up_name = format!("layer_{layer_idx}.ffn.up");
        let up_ptr = executor.get_quantized_weight_ptr(&up_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{up_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            up_ptr,
            input,
            &up_gpu,
            intermediate_dim as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Up GEMV failed: {e}") })?;

        // SwiGLU activation (CPU - fusing requires custom kernel)
        let gate = gpu_to_host(&gate_gpu)?;
        let up = gpu_to_host(&up_gpu)?;

        let activated: Vec<f32> = gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        // Down projection
        let activated_gpu = GpuBuffer::from_host(executor.context(), &activated)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload activated: {e}") })?;

        let down_gpu = GpuBuffer::new(executor.context(), hidden_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate down: {e}") })?;

        let down_name = format!("layer_{layer_idx}.ffn.down");
        let down_ptr = executor.get_quantized_weight_ptr(&down_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{down_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            down_ptr,
            &activated_gpu,
            &down_gpu,
            hidden_dim as u32,
            intermediate_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Down GEMV failed: {e}") })?;

        Ok(down_gpu)
    }

    /// Standard FFN (GELU) using Q4_0 kernels
    #[cfg(feature = "cuda")]
    fn ffn_standard_gpu(
        &self,
        executor: &mut CudaExecutor,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Up projection
        let up_gpu = GpuBuffer::new(executor.context(), intermediate_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate up: {e}") })?;

        let up_name = format!("layer_{layer_idx}.ffn.up");
        let up_ptr = executor.get_quantized_weight_ptr(&up_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{up_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            up_ptr,
            input,
            &up_gpu,
            intermediate_dim as u32,
            hidden_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Up GEMV failed: {e}") })?;

        // GELU activation (CPU)
        let up = gpu_to_host(&up_gpu)?;

        let activated: Vec<f32> = up.iter().map(|&x| gelu(x)).collect();

        // Down projection
        let activated_gpu = GpuBuffer::from_host(executor.context(), &activated)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to upload activated: {e}") })?;

        let down_gpu = GpuBuffer::new(executor.context(), hidden_dim)
            .map_err(|e| RealizarError::GpuError { reason: format!("Failed to allocate down: {e}") })?;

        let down_name = format!("layer_{layer_idx}.ffn.down");
        let down_ptr = executor.get_quantized_weight_ptr(&down_name)
            .map_err(|e| RealizarError::GpuError { reason: format!("{down_name} not cached: {e}") })?;

        executor.q4_0_gemv_into(
            down_ptr,
            &activated_gpu,
            &down_gpu,
            hidden_dim as u32,
            intermediate_dim as u32,
        ).map_err(|e| RealizarError::GpuError { reason: format!("Down GEMV failed: {e}") })?;

        Ok(down_gpu)
    }

    /// RMSNorm in place
    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32]) {
        let eps = self.config.eps;
        let n = x.len();

        // Calculate RMS
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let scale = 1.0 / rms;

        // Normalize and apply weight
        for (i, v) in x.iter_mut().enumerate() {
            *v = *v * scale * weight.get(i).copied().unwrap_or(1.0);
        }
    }

    /// Apply RoPE to Q and K within fused QKV tensor
    ///
    /// # Arguments
    ///
    /// * `qkv` - Fused QKV tensor [Q | K | V] where Q is [seq_len * hidden_dim]
    /// * `seq_len` - Number of tokens
    /// * `hidden_dim` - Q dimension
    /// * `num_heads` - Number of Q heads
    /// * `num_kv_heads` - Number of KV heads (for GQA)
    fn apply_rope_to_qkv(
        &self,
        qkv: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;
        let theta = self.config.rope_theta;

        for pos in 0..seq_len {
            let qkv_start = pos * qkv_dim;

            // Apply RoPE to Q
            let q_start = qkv_start;
            self.apply_rope_inplace(&mut qkv[q_start..q_start + hidden_dim], pos, num_heads, head_dim, theta);

            // Apply RoPE to K
            let k_start = qkv_start + hidden_dim;
            self.apply_rope_inplace(&mut qkv[k_start..k_start + kv_dim], pos, num_kv_heads, head_dim, theta);
        }
    }

    /// Apply RoPE to a single Q or K tensor at given position
    fn apply_rope_inplace(
        &self,
        x: &mut [f32],
        position: usize,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
    ) {
        let half_dim = head_dim / 2;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        for h in 0..num_heads {
            let head_start = h * head_dim;

            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let idx1 = head_start + i;
                let idx2 = head_start + half_dim + i;

                if idx2 < x.len() {
                    let x1 = x[idx1];
                    let x2 = x[idx2];

                    x[idx1] = x1 * cos_val - x2 * sin_val;
                    x[idx2] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    /// Simple attention (CPU, single-token)
    fn attention_cpu(
        &self,
        qkv: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Vec<f32> {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Split QKV
        let _q = &qkv[..hidden_dim];
        let _k = &qkv[hidden_dim..hidden_dim + kv_dim];
        let v = &qkv[hidden_dim + kv_dim..];

        // For single token, attention is trivial (softmax of single score = 1.0)
        // Output is just V projected back
        if seq_len == 1 {
            // GQA: repeat KV heads to match Q heads
            let kv_repeat = num_heads / num_kv_heads;
            let mut out = vec![0.0; hidden_dim];

            for h in 0..num_heads {
                let kv_h = h / kv_repeat;
                for d in 0..head_dim {
                    out[h * head_dim + d] = v[kv_h * head_dim + d];
                }
            }

            out
        } else {
            // Multi-token attention (simplified)
            // In practice, we'd use KV cache here
            v[..hidden_dim.min(v.len())].to_vec()
        }
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (tanh approximation)
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_basic() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.5); // SiLU(1) ≈ 0.731
        assert!(silu(-1.0) < 0.0); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_gelu_basic() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8); // GELU(1) ≈ 0.841
        assert!(gelu(-1.0) < 0.0); // GELU(-1) ≈ -0.159
    }

    #[test]
    fn test_gpu_model_q4_config() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "llama".to_string(),
                hidden_dim: 2048,
                num_layers: 22,
                num_heads: 32,
                num_kv_heads: 4,
                vocab_size: 32000,
                intermediate_dim: 5632,
                context_length: 2048,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.0; 32000 * 2048],
            output_norm_weight: vec![1.0; 2048],
            layer_norms: vec![LayerNorms {
                attn_norm: vec![1.0; 2048],
                ffn_norm: vec![1.0; 2048],
            }; 22],
            num_layers: 22,
            has_gate: true,
        };

        assert_eq!(model.num_layers, 22);
        assert!(model.has_gate);
        assert_eq!(model.config.hidden_dim, 2048);
    }

    #[test]
    fn test_rms_norm() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 4,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                vocab_size: 100,
                intermediate_dim: 8,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.0; 400],
            output_norm_weight: vec![1.0; 4],
            layer_norms: vec![LayerNorms {
                attn_norm: vec![1.0; 4],
                ffn_norm: vec![1.0; 4],
            }],
            num_layers: 1,
            has_gate: false,
        };

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        model.rms_norm_inplace(&mut x, &weight);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // Normalized: [0.365, 0.730, 1.095, 1.461]
        let rms = (30.0_f32 / 4.0).sqrt();
        assert!((x[0] - 1.0 / rms).abs() < 1e-4);
        assert!((x[1] - 2.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_attention_single_token() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 8,
                num_layers: 1,
                num_heads: 2,
                num_kv_heads: 2,
                vocab_size: 100,
                intermediate_dim: 16,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.0; 800],
            output_norm_weight: vec![1.0; 8],
            layer_norms: vec![LayerNorms {
                attn_norm: vec![1.0; 8],
                ffn_norm: vec![1.0; 8],
            }],
            num_layers: 1,
            has_gate: false,
        };

        // QKV: Q[8] + K[8] + V[8] = 24
        let qkv = vec![
            // Q
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            // K
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            // V
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];

        let out = model.attention_cpu(&qkv, 1, 8, 2, 2);

        // For single token, output = V (all scores softmax to 1.0)
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layer_norms_creation() {
        let norms = LayerNorms {
            attn_norm: vec![1.0; 128],
            ffn_norm: vec![1.0; 128],
        };

        assert_eq!(norms.attn_norm.len(), 128);
        assert_eq!(norms.ffn_norm.len(), 128);
    }

    #[test]
    fn test_adapter_create_model() {
        use crate::apr_transformer::{QuantizedAprLayerQ4, QuantizedAprTensorQ4};

        let apr = QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 256,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 1000,
                intermediate_dim: 512,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.0; 1000 * 256],
            layers: vec![
                QuantizedAprLayerQ4 {
                    attn_norm_weight: vec![1.0; 256],
                    qkv_weight: QuantizedAprTensorQ4::zeros(256, 256 * 3),
                    attn_output_weight: QuantizedAprTensorQ4::zeros(256, 256),
                    ffn_up_weight: QuantizedAprTensorQ4::zeros(256, 512),
                    ffn_down_weight: QuantizedAprTensorQ4::zeros(512, 256),
                    ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(256, 512)),
                    ffn_norm_weight: Some(vec![1.0; 256]),
                },
                QuantizedAprLayerQ4 {
                    attn_norm_weight: vec![1.0; 256],
                    qkv_weight: QuantizedAprTensorQ4::zeros(256, 256 * 3),
                    attn_output_weight: QuantizedAprTensorQ4::zeros(256, 256),
                    ffn_up_weight: QuantizedAprTensorQ4::zeros(256, 512),
                    ffn_down_weight: QuantizedAprTensorQ4::zeros(512, 256),
                    ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(256, 512)),
                    ffn_norm_weight: Some(vec![1.0; 256]),
                },
            ],
            output_norm_weight: vec![1.0; 256],
            lm_head_weight: QuantizedAprTensorQ4::zeros(256, 1000),
        };

        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.num_layers, 2);
        assert!(model.has_gate);
        assert_eq!(model.layer_norms.len(), 2);
        assert_eq!(model.token_embedding.len(), 256000);
    }
}
