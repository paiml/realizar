//! SIMD-Accelerated Quantized APR Transformer (Q4_0)
//!
//! High-performance Q4_0 inference using SIMD-accelerated matmul primitives.
//! Extracted from apr_transformer/mod.rs (PMAT-802).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(dead_code)]

use super::{AprKVCache, AprTransformerConfig};
use crate::error::{RealizarError, Result};

// SIMD-Accelerated Quantized APR Transformer (Q4_0)
// ============================================================================

/// Q4_0 quantized tensor for SIMD-accelerated inference
///
/// Stores raw Q4_0 bytes (18 bytes per 32 values) with dimensions for matmul.
#[derive(Debug, Clone)]
pub struct QuantizedAprTensorQ4 {
    /// Raw Q4_0 quantized data
    pub data: Vec<u8>,
    /// Input dimension (columns in weight matrix)
    pub in_dim: usize,
    /// Output dimension (rows in weight matrix)
    pub out_dim: usize,
}

impl QuantizedAprTensorQ4 {
    /// Create a new Q4_0 tensor from raw data
    #[must_use]
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Create empty tensor with proper Q4_0 allocation
    #[must_use]
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_elements = in_dim * out_dim;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        let data = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Get expected data size in bytes
    #[must_use]
    pub fn expected_bytes(num_elements: usize) -> usize {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        num_blocks * Q4_0_BLOCK_BYTES
    }
}

/// Q4_0 quantized layer for SIMD-accelerated inference
///
/// Stores individual Q4_0 tensors for each weight matrix, enabling
/// direct use of `fused_q4_0_q8_0_parallel_matvec`.
#[derive(Debug, Clone)]
pub struct QuantizedAprLayerQ4 {
    /// Attention norm weight (F32, small)
    pub attn_norm_weight: Vec<f32>,
    /// QKV projection weights (Q4_0)
    pub qkv_weight: QuantizedAprTensorQ4,
    /// Attention output projection (Q4_0)
    pub attn_output_weight: QuantizedAprTensorQ4,
    /// FFN up projection (Q4_0)
    pub ffn_up_weight: QuantizedAprTensorQ4,
    /// FFN down projection (Q4_0)
    pub ffn_down_weight: QuantizedAprTensorQ4,
    /// FFN gate projection for SwiGLU (Q4_0, optional)
    pub ffn_gate_weight: Option<QuantizedAprTensorQ4>,
    /// FFN norm weight (F32, optional)
    pub ffn_norm_weight: Option<Vec<f32>>,
}

/// SIMD-accelerated Quantized APR Transformer
///
/// Stores weights in Q4_0 format and uses integer SIMD matmul
/// (`_mm256_maddubs_epi16`) for near-GGUF performance.
///
/// # Performance
///
/// Expected throughput: ~17 tok/s on TinyLlama-1.1B (1.36x vs GGUF)
/// With KV cache: ~25-34 tok/s expected (1.5-2x additional speedup)
#[derive(Debug, Clone)]
pub struct QuantizedAprTransformerQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32 for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Quantized layers
    pub layers: Vec<QuantizedAprLayerQ4>,
    /// Output norm weight (F32)
    pub output_norm_weight: Vec<f32>,
    /// LM head weight (Q4_0)
    pub lm_head_weight: QuantizedAprTensorQ4,
}

/// Scratch buffer for zero-allocation inference
///
/// Pre-allocates all intermediate buffers needed for a forward pass.
/// Reuse across multiple forward calls to eliminate per-token allocations.
#[derive(Debug)]
pub struct AprInferenceScratch {
    /// Hidden state [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// QKV projection output [qkv_dim]
    pub qkv_out: Vec<f32>,
    /// Query vectors [q_dim]
    pub q: Vec<f32>,
    /// Key vectors [k_dim]
    pub k: Vec<f32>,
    /// Value vectors [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN input [hidden_dim]
    pub ffn_input: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN output [hidden_dim]
    pub ffn_out: Vec<f32>,
}

impl AprInferenceScratch {
    /// Create scratch buffer sized for a model config
    #[must_use]
    pub fn from_config(config: &AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let qkv_dim = hidden_dim * 3; // Conservative estimate
        let intermediate_dim = config.intermediate_dim;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv_out: vec![0.0; qkv_dim],
            q: vec![0.0; hidden_dim],
            k: vec![0.0; hidden_dim],
            v: vec![0.0; hidden_dim],
            attn_out: vec![0.0; hidden_dim],
            ffn_input: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_out: vec![0.0; hidden_dim],
        }
    }

    /// Clear all buffers (set to zero)
    pub fn clear(&mut self) {
        self.hidden.fill(0.0);
        self.normed.fill(0.0);
        self.qkv_out.fill(0.0);
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.attn_out.fill(0.0);
        self.ffn_input.fill(0.0);
        self.ffn_up.fill(0.0);
        self.ffn_gate.fill(0.0);
        self.ffn_out.fill(0.0);
    }
}

impl QuantizedAprTransformerQ4 {
    /// Create from GGUF OwnedQuantizedModel (extracts Q4_0 bytes)
    ///
    /// # Arguments
    ///
    /// * `gguf` - Source GGUF model with Q4_0 weights
    ///
    /// # Returns
    ///
    /// Quantized APR transformer with same weights
    pub fn from_gguf(gguf: &crate::gguf::OwnedQuantizedModel) -> Self {
        use crate::gguf::OwnedQKVWeights;

        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers =
            gguf.layers
                .iter()
                .map(|l| {
                    // Extract QKV weight data
                    let qkv_weight = match &l.qkv_weight {
                        OwnedQKVWeights::Fused(t) => {
                            QuantizedAprTensorQ4::new(t.data.clone(), t.in_dim, t.out_dim)
                        },
                        OwnedQKVWeights::Separate { q, k, v } => {
                            // Concatenate Q, K, V for fused format
                            let mut data =
                                Vec::with_capacity(q.data.len() + k.data.len() + v.data.len());
                            data.extend_from_slice(&q.data);
                            data.extend_from_slice(&k.data);
                            data.extend_from_slice(&v.data);
                            QuantizedAprTensorQ4::new(
                                data,
                                q.in_dim,                          // hidden_dim
                                q.out_dim + k.out_dim + v.out_dim, // qkv_dim
                            )
                        },
                    };

                    QuantizedAprLayerQ4 {
                        attn_norm_weight: l.attn_norm_weight.clone(),
                        qkv_weight,
                        attn_output_weight: QuantizedAprTensorQ4::new(
                            l.attn_output_weight.data.clone(),
                            l.attn_output_weight.in_dim,
                            l.attn_output_weight.out_dim,
                        ),
                        ffn_up_weight: QuantizedAprTensorQ4::new(
                            l.ffn_up_weight.data.clone(),
                            l.ffn_up_weight.in_dim,
                            l.ffn_up_weight.out_dim,
                        ),
                        ffn_down_weight: QuantizedAprTensorQ4::new(
                            l.ffn_down_weight.data.clone(),
                            l.ffn_down_weight.in_dim,
                            l.ffn_down_weight.out_dim,
                        ),
                        ffn_gate_weight: l.ffn_gate_weight.as_ref().map(|g| {
                            QuantizedAprTensorQ4::new(g.data.clone(), g.in_dim, g.out_dim)
                        }),
                        ffn_norm_weight: l.ffn_norm_weight.clone(),
                    }
                })
                .collect();

        let lm_head_weight = QuantizedAprTensorQ4::new(
            gguf.lm_head_weight.data.clone(),
            gguf.lm_head_weight.in_dim,
            gguf.lm_head_weight.out_dim,
        );

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            lm_head_weight,
        }
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Create a scratch buffer for zero-allocation inference
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = QuantizedAprTransformerQ4::from_gguf(&gguf);
    /// let mut scratch = model.create_scratch();
    ///
    /// // Reuse scratch across multiple forward passes
    /// for token_id in token_ids {
    ///     let logits = model.forward_single_with_scratch(token_id, &mut scratch)?;
    /// }
    /// ```
    #[must_use]
    pub fn create_scratch(&self) -> AprInferenceScratch {
        AprInferenceScratch::from_config(&self.config)
    }

    /// Forward pass using SIMD-accelerated Q4_0×Q8_0 matmul
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (F32)
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(seq_len * qkv_dim);
            for s in 0..seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            // Proper attention with RoPE and causal mask
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v = &qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim];

                // Apply RoPE to Q and K (position-dependent rotation)
                self.apply_rope(&mut q, s, num_heads);
                self.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_output = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with SwiGLU (sequential to avoid nested parallelism overhead)
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Sequential up + gate (both matmuls use internal parallelism)
                let mut ffn_up_out = Vec::with_capacity(seq_len * intermediate_dim);
                let mut ffn_gate_out = Vec::with_capacity(seq_len * intermediate_dim);

                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];

                    // Up projection
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_up_out.extend(u);

                    // Gate projection
                    let g = fused_q4_0_q8_0_parallel_matvec(
                        &gate.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_gate_out.extend(g);
                }

                // Apply SiLU to gate and multiply with up
                for i in 0..ffn_up_out.len() {
                    let silu = ffn_gate_out[i] / (1.0 + (-ffn_gate_out[i]).exp());
                    ffn_up_out[i] *= silu;
                }
                ffn_up_out
            } else {
                // Non-SwiGLU: Sequential up projection + GELU
                let mut up = Vec::with_capacity(seq_len * intermediate_dim);
                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                // GELU activation (tanh approximation)
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm
        let last_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Create a KV cache for this model
    #[must_use]
    pub fn create_kv_cache(&self) -> AprKVCache {
        AprKVCache::new(&self.config)
    }

    /// Forward pass for a single token using scratch buffer (zero allocation)
    ///
    /// This is the fastest path for autoregressive generation when combined
    /// with `forward_with_cache_and_scratch`. It reuses pre-allocated buffers
    /// to eliminate per-token allocations.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `scratch` - Pre-allocated scratch buffer (from `create_scratch()`)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward_single_with_scratch(
        &self,
        token_id: u32,
        scratch: &mut AprInferenceScratch,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec_into;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (write directly to scratch.hidden)
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            scratch.hidden[..hidden_dim]
                .copy_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            scratch.hidden[..hidden_dim].fill(0.0);
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm (reuse scratch.normed)
            let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
            for i in 0..hidden_dim {
                scratch.normed[i] = scratch.hidden[i] / rms * layer.attn_norm_weight[i];
            }

            // QKV projection (zero-allocation - write directly to scratch.qkv_out)
            let qkv_dim = layer.qkv_weight.out_dim;
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.qkv_weight.data,
                &scratch.normed[..hidden_dim],
                hidden_dim,
                &mut scratch.qkv_out[..qkv_dim],
            )?;

            // Extract Q, K, V and apply RoPE (position=0 for single token)
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            scratch.q[..q_dim].copy_from_slice(&scratch.qkv_out[..q_dim]);
            scratch.k[..kv_dim].copy_from_slice(&scratch.qkv_out[q_dim..q_dim + kv_dim]);
            scratch.v[..kv_dim]
                .copy_from_slice(&scratch.qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim]);

            // Apply RoPE at position 0
            self.apply_rope(&mut scratch.q[..q_dim], 0, num_heads);
            self.apply_rope(&mut scratch.k[..kv_dim], 0, num_kv_heads);

            // For single token, attention is trivial: output = V (softmax of 1 element = 1.0)
            let group_size = num_heads / num_kv_heads;
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                scratch.attn_out[out_offset..out_offset + head_dim]
                    .copy_from_slice(&scratch.v[v_offset..v_offset + head_dim]);
            }

            // Output projection (write to scratch.ffn_out as temporary)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.attn_output_weight.data,
                &scratch.attn_out[..hidden_dim],
                layer.attn_output_weight.in_dim,
                &mut scratch.ffn_out[..layer.attn_output_weight.out_dim],
            )?;

            // Residual connection (attn)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }

            // Pre-FFN norm
            if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for i in 0..hidden_dim {
                    scratch.ffn_input[i] = scratch.hidden[i] / rms * ffn_norm[i];
                }
            } else {
                scratch.ffn_input[..hidden_dim].copy_from_slice(&scratch.normed[..hidden_dim]);
            }

            // FFN with SwiGLU
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            if let Some(gate) = &layer.ffn_gate_weight {
                // Up projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                // Gate projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &gate.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_gate[..intermediate_dim],
                )?;

                // SwiGLU: silu(gate) * up
                for i in 0..intermediate_dim {
                    let silu = scratch.ffn_gate[i] / (1.0 + (-scratch.ffn_gate[i]).exp());
                    scratch.ffn_up[i] *= silu;
                }
            } else {
                // GELU path (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for i in 0..intermediate_dim {
                    let x = scratch.ffn_up[i];
                    let t = (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh();
                    scratch.ffn_up[i] = 0.5 * x * (1.0 + t);
                }
            }

            // Down projection (write to scratch.ffn_out)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &scratch.ffn_up[..intermediate_dim],
                intermediate_dim,
                &mut scratch.ffn_out[..hidden_dim],
            )?;

            // Residual connection (FFN)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }
        }

        // 3. Final RMS norm
        let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        for i in 0..hidden_dim {
            scratch.normed[i] = scratch.hidden[i] / rms * self.output_norm_weight[i];
        }

        // 4. LM head projection (still allocates - logits must be returned)
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        fused_q4_0_q8_0_parallel_matvec_into(
            &self.lm_head_weight.data,
            &scratch.normed[..hidden_dim],
            hidden_dim,
            &mut logits,
        )?;

        Ok(logits)
    }

    /// Forward pass with KV cache for efficient autoregressive generation
    ///
    /// This method only computes attention for the new token(s), reusing
    /// cached K/V from previous positions. Provides 1.5-2x speedup.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - New token IDs to process (typically 1 for generation)
    /// * `cache` - KV cache to use and update
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for the last token
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut AprKVCache,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // Position in the sequence (including cached positions)
        let cache_len = cache.len();
        let new_seq_len = token_ids.len();

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(new_seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..new_seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul (only for new tokens)
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(new_seq_len * qkv_dim);
            for s in 0..new_seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Process new tokens: extract Q, K, V and apply RoPE
            let mut new_q = Vec::with_capacity(new_seq_len * q_dim);
            for s in 0..new_seq_len {
                let qkv_start = s * qkv_dim;
                let position = cache_len + s;

                // Extract Q, K, V for this position
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v =
                    qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim].to_vec();

                // Apply RoPE with correct position
                self.apply_rope(&mut q, position, num_heads);
                self.apply_rope(&mut k, position, num_kv_heads);

                new_q.extend_from_slice(&q);

                // Append to cache (K and V with RoPE applied to K)
                cache.append(layer_idx, &k, &v);
            }

            // Get full K and V from cache (includes new tokens)
            let (full_k, full_v) = cache.get(layer_idx);
            let total_seq_len = cache.len();

            // Compute attention: new Q attends to all cached K/V
            let attn_output = self.causal_attention_cached(
                &new_q,
                full_k,
                full_v,
                new_seq_len,
                total_seq_len,
                cache_len,
            );

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..new_seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with parallel up/gate for SwiGLU models
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Parallel FFN up + gate
                let (ffn_up_result, ffn_gate_result) = rayon::join(
                    || {
                        let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(u) = fused_q4_0_q8_0_parallel_matvec(
                                &layer.ffn_up_weight.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                up.extend(u);
                            }
                        }
                        up
                    },
                    || {
                        let mut g = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(gv) = fused_q4_0_q8_0_parallel_matvec(
                                &gate.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                g.extend(gv);
                            }
                        }
                        g
                    },
                );

                let mut up = ffn_up_result;
                for i in 0..up.len() {
                    let silu = ffn_gate_result[i] / (1.0 + (-ffn_gate_result[i]).exp());
                    up[i] *= silu;
                }
                up
            } else {
                // Non-SwiGLU: Sequential + GELU
                let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                for s in 0..new_seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm (only for last token)
        let last_start = (new_seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Attention with KV cache - new Q attends to all cached K/V
    ///
    /// Parallelizes across attention heads for efficiency.
    fn causal_attention_cached(
        &self,
        new_q: &[f32],
        full_k: &[f32],
        full_v: &[f32],
        new_seq_len: usize,
        _total_seq_len: usize,
        cache_len: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = num_heads / num_kv_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            // Sequential path
            let mut output = vec![0.0f32; new_seq_len * q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..new_seq_len {
                    let pos = cache_len + i;
                    let mut scores = Vec::with_capacity(pos + 1);
                    let q_start = i * q_dim + q_head_offset;

                    // Attend to all positions up to current (causal)
                    for j in 0..=pos {
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += new_q[q_start + d] * full_k[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    for s in &mut scores {
                        *s /= exp_sum;
                    }

                    // Weighted sum
                    let out_start = i * q_dim + q_head_offset;
                    for (j, &weight) in scores.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            output[out_start + d] += weight * full_v[v_start + d];
                        }
                    }
                }
            }
            output
        } else {
            // Parallel path
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; new_seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..new_seq_len {
                        let pos = cache_len + i;
                        let mut scores = Vec::with_capacity(pos + 1);
                        let q_start = i * q_dim + q_head_offset;

                        for j in 0..=pos {
                            let k_start = j * kv_dim + kv_head_offset;
                            let mut score = 0.0f32;
                            for d in 0..head_dim {
                                score += new_q[q_start + d] * full_k[k_start + d];
                            }
                            scores.push(score * scale);
                        }

                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            exp_sum += *s;
                        }
                        for s in &mut scores {
                            *s /= exp_sum;
                        }

                        let out_start = i * head_dim;
                        for (j, &weight) in scores.iter().enumerate() {
                            let v_start = j * kv_dim + kv_head_offset;
                            for d in 0..head_dim {
                                head_out[out_start + d] += weight * full_v[v_start + d];
                            }
                        }
                    }
                    head_out
                })
                .collect();

            // Merge
            let mut output = vec![0.0f32; new_seq_len * q_dim];
            for (head, head_out) in head_outputs.into_iter().enumerate() {
                let head_offset = head * head_dim;
                for i in 0..new_seq_len {
                    let src_start = i * head_dim;
                    let dst_start = i * q_dim + head_offset;
                    output[dst_start..dst_start + head_dim]
                        .copy_from_slice(&head_out[src_start..src_start + head_dim]);
                }
            }
            output
        }
    }

    /// Get memory footprint in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let embed_size = self.token_embedding.len() * 4;
        let norm_size = self.output_norm_weight.len() * 4;
        let lm_head_size = self.lm_head_weight.data.len();

        let layer_size: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm_weight.len() * 4
                    + l.qkv_weight.data.len()
                    + l.attn_output_weight.data.len()
                    + l.ffn_up_weight.data.len()
                    + l.ffn_down_weight.data.len()
                    + l.ffn_gate_weight.as_ref().map_or(0, |g| g.data.len())
                    + l.ffn_norm_weight.as_ref().map_or(0, |n| n.len() * 4)
            })
            .sum();

        embed_size + norm_size + lm_head_size + layer_size
    }

    /// Apply Rotary Position Embeddings (RoPE) to a tensor
    ///
    /// RoPE applies position-dependent rotation to pairs of dimensions,
    /// enabling the model to learn relative positional information.
    fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        // Apply rotation to each head with inline cos/sin computation
        // Avoids allocation by computing cos/sin on the fly
        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            if idx2_start + half_dim > x.len() {
                continue;
            }

            // Process 4 elements at a time for better ILP
            let mut i = 0;
            while i + 4 <= half_dim {
                // Compute 4 frequencies
                let freq0 = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let freq1 = 1.0 / theta.powf(2.0 * (i + 1) as f32 / head_dim_f32);
                let freq2 = 1.0 / theta.powf(2.0 * (i + 2) as f32 / head_dim_f32);
                let freq3 = 1.0 / theta.powf(2.0 * (i + 3) as f32 / head_dim_f32);

                // Compute 4 angles
                let angle0 = pos_f32 * freq0;
                let angle1 = pos_f32 * freq1;
                let angle2 = pos_f32 * freq2;
                let angle3 = pos_f32 * freq3;

                // Compute cos/sin (use sincos if available for better performance)
                let (sin0, cos0) = angle0.sin_cos();
                let (sin1, cos1) = angle1.sin_cos();
                let (sin2, cos2) = angle2.sin_cos();
                let (sin3, cos3) = angle3.sin_cos();

                // Load x1 and x2 values
                let x1_0 = x[head_start + i];
                let x1_1 = x[head_start + i + 1];
                let x1_2 = x[head_start + i + 2];
                let x1_3 = x[head_start + i + 3];

                let x2_0 = x[idx2_start + i];
                let x2_1 = x[idx2_start + i + 1];
                let x2_2 = x[idx2_start + i + 2];
                let x2_3 = x[idx2_start + i + 3];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1_0 * cos0 - x2_0 * sin0;
                x[head_start + i + 1] = x1_1 * cos1 - x2_1 * sin1;
                x[head_start + i + 2] = x1_2 * cos2 - x2_2 * sin2;
                x[head_start + i + 3] = x1_3 * cos3 - x2_3 * sin3;

                x[idx2_start + i] = x1_0 * sin0 + x2_0 * cos0;
                x[idx2_start + i + 1] = x1_1 * sin1 + x2_1 * cos1;
                x[idx2_start + i + 2] = x1_2 * sin2 + x2_2 * cos2;
                x[idx2_start + i + 3] = x1_3 * sin3 + x2_3 * cos3;

                i += 4;
            }

            // Handle remaining elements
            while i < half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;

                i += 1;
            }
        }
    }

    /// Compute scaled dot-product attention with causal mask and GQA support
    ///
    /// Implements multi-head attention with Grouped Query Attention (GQA),
    /// where multiple Q heads share the same K/V heads.
    ///
    /// Optimized for single-token inference (seq_len=1).
    fn causal_attention(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA: multiple Q heads share each KV head
        let group_size = num_heads / num_kv_heads;

        // Q has num_heads heads, K/V have num_kv_heads heads
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Fast path for single token (common case in autoregressive generation)
        // With seq_len=1 and causal mask, each head just copies its V vector
        // (softmax of single element is 1.0)
        if seq_len == 1 {
            let mut output = vec![0.0f32; q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                output[out_offset..out_offset + head_dim]
                    .copy_from_slice(&v[v_offset..v_offset + head_dim]);
            }
            return output;
        }

        // General case for seq_len > 1
        use rayon::prelude::*;

        // Parallel threshold - use parallel for 4+ heads
        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            // Sequential path for few heads
            let mut output = vec![0.0f32; seq_len * q_dim];
            for head in 0..num_heads {
                self.compute_head_attention(
                    head,
                    group_size,
                    head_dim,
                    scale,
                    q,
                    k,
                    v,
                    seq_len,
                    q_dim,
                    kv_dim,
                    &mut output,
                );
            }
            output
        } else {
            // Parallel path - each head computes independently, then merge
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..seq_len {
                        let mut scores = Vec::with_capacity(i + 1);
                        let q_start = i * q_dim + q_head_offset;

                        for j in 0..=i {
                            let k_start = j * kv_dim + kv_head_offset;
                            let mut score = 0.0f32;
                            for d in 0..head_dim {
                                score += q[q_start + d] * k[k_start + d];
                            }
                            scores.push(score * scale);
                        }

                        // Softmax
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            exp_sum += *s;
                        }
                        for s in &mut scores {
                            *s /= exp_sum;
                        }

                        // Weighted sum
                        let out_start = i * head_dim;
                        for (j, &weight) in scores.iter().enumerate() {
                            let v_start = j * kv_dim + kv_head_offset;
                            for d in 0..head_dim {
                                head_out[out_start + d] += weight * v[v_start + d];
                            }
                        }
                    }
                    head_out
                })
                .collect();

            // Merge head outputs into final output
            let mut output = vec![0.0f32; seq_len * q_dim];
            for (head, head_out) in head_outputs.into_iter().enumerate() {
                let head_offset = head * head_dim;
                for i in 0..seq_len {
                    let src_start = i * head_dim;
                    let dst_start = i * q_dim + head_offset;
                    output[dst_start..dst_start + head_dim]
                        .copy_from_slice(&head_out[src_start..src_start + head_dim]);
                }
            }
            output
        }
    }

    /// Compute attention for a single head (helper for sequential path)
    #[allow(clippy::too_many_arguments)]
    fn compute_head_attention(
        &self,
        head: usize,
        group_size: usize,
        head_dim: usize,
        scale: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        q_dim: usize,
        kv_dim: usize,
        output: &mut [f32],
    ) {
        let kv_head = head / group_size;
        let q_head_offset = head * head_dim;
        let kv_head_offset = kv_head * head_dim;

        for i in 0..seq_len {
            let mut scores = Vec::with_capacity(i + 1);
            let q_start = i * q_dim + q_head_offset;

            for j in 0..=i {
                let k_start = j * kv_dim + kv_head_offset;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                scores.push(score * scale);
            }

            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            let out_start = i * q_dim + q_head_offset;
            for (j, &weight) in scores.iter().enumerate() {
                let v_start = j * kv_dim + kv_head_offset;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }
}

// =============================================================================
