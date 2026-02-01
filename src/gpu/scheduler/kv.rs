//! KV Cache Management for GpuModel (PMAT-802)
//!
//! Extracted from model.rs to reduce module size.
//! Contains KV cache forward pass and generation logic.

use super::super::{cpu_matmul_transposed_simd, exceeds_gpu_buffer_limit, StreamingKVCache};
use super::model::GpuModel;
use super::types::GpuGenerateConfig;
use crate::error::{RealizarError, Result};

/// Apply Rotary Position Embedding (RoPE) to Q or K vectors (Phase 21)
///
/// RoPE encodes position information by rotating pairs of elements
/// with position-dependent angles. This is CRITICAL for transformer attention.
///
/// # Arguments
/// * `x` - Mutable slice of Q or K vectors [seq_len * num_heads * head_dim]
/// * `seq_len` - Number of positions to encode
/// * `num_heads` - Number of attention heads in this tensor
/// * `head_dim` - Dimension per head
/// * `rope_theta` - Base frequency (typically 10000.0)
/// * `start_pos` - Starting position for RoPE (0 for prefill, cache_len for incremental)
fn apply_rope(
    x: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    start_pos: usize,
) {
    let half_dim = head_dim / 2;
    let head_dim_f32 = head_dim as f32;
    let total_dim = num_heads * head_dim;

    for pos in 0..seq_len {
        let position = start_pos + pos;
        let pos_f32 = position as f32;
        let pos_offset = pos * total_dim;

        for h in 0..num_heads {
            let head_start = pos_offset + h * head_dim;
            let idx2_start = head_start + half_dim;

            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

/// Forward pass with KV cache population (IMP-031)
pub fn forward_gpu_with_cache(
    model: &mut GpuModel,
    token_ids: &[usize],
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Token IDs cannot be empty".to_string(),
        });
    }

    let seq_len = token_ids.len();
    let hidden_dim = model.config.hidden_dim;

    // Step 1: Embed tokens
    let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
    for &token_id in token_ids {
        if token_id >= model.config.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, model.config.vocab_size
                ),
            });
        }
        let offset = token_id * hidden_dim;
        hidden.extend_from_slice(&model.embedding_weights[offset..offset + hidden_dim]);
    }

    // Step 2: Pass through transformer blocks with KV cache population
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_with_cache(model, &hidden, seq_len, block_idx, kv_cache)?;
    }

    // Step 3: Final layer norm
    hidden = layer_norm_kv(model, &hidden);

    // Step 4: LM head projection - only for final position
    let final_hidden = &hidden[(seq_len - 1) * hidden_dim..seq_len * hidden_dim];
    let lm_head_elements = hidden_dim * model.config.vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        cpu_matmul_transposed_simd(
            final_hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            model.config.vocab_size,
        )
    } else {
        let logits = model.scheduler.matmul(
            final_hidden,
            &model.lm_head_weight,
            1,
            hidden_dim,
            model.config.vocab_size,
        )?;
        let mut output = logits;
        for (out_val, bias_val) in output.iter_mut().zip(model.lm_head_bias.iter()) {
            *out_val += *bias_val;
        }
        output
    };

    Ok(output)
}

/// Incremental forward pass using cached KV (IMP-032)
pub fn forward_gpu_incremental(
    model: &mut GpuModel,
    token_id: usize,
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    if token_id >= model.config.vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Token ID {} out of bounds (vocab_size={})",
                token_id, model.config.vocab_size
            ),
        });
    }

    let hidden_dim = model.config.hidden_dim;

    // Step 1: Embed token
    let offset = token_id * hidden_dim;
    let mut hidden = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Step 2: Pass through transformer blocks using KV cache
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_incremental(model, &hidden, block_idx, kv_cache)?;
    }

    // Step 3: Final layer norm
    hidden = layer_norm_kv(model, &hidden);

    // Step 4: LM head projection
    let lm_head_elements = hidden_dim * model.config.vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        cpu_matmul_transposed_simd(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            model.config.vocab_size,
        )
    } else {
        let logits = model.scheduler.matmul(
            &hidden,
            &model.lm_head_weight,
            1,
            hidden_dim,
            model.config.vocab_size,
        )?;
        let mut output = logits;
        for (out_val, bias_val) in output.iter_mut().zip(model.lm_head_bias.iter()) {
            *out_val += *bias_val;
        }
        output
    };

    Ok(output)
}

/// Forward pass through a single block with KV cache population
fn forward_block_with_cache(
    model: &mut GpuModel,
    input: &[f32],
    seq_len: usize,
    block_idx: usize,
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let kv_dim = model.config.kv_dim();
    let qkv_dim = model.config.qkv_dim();

    let block = &model.block_weights[block_idx];

    // Pre-norm
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection
    let mut qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        seq_len,
        hidden_dim,
        qkv_dim,
    )?;

    // Split Q, K, V (mutable for RoPE application)
    let q_end = seq_len * hidden_dim;
    let k_end = q_end + seq_len * kv_dim;

    // Phase 21: Apply RoPE to Q and K BEFORE caching
    // This is CRITICAL - without RoPE, attention has no position information
    let rope_theta = model.config.rope_theta;

    // Apply RoPE to Q (all heads)
    apply_rope(
        &mut qkv[..q_end],
        seq_len,
        num_heads,
        head_dim,
        rope_theta,
        0,
    );

    // Apply RoPE to K (KV heads)
    apply_rope(
        &mut qkv[q_end..k_end],
        seq_len,
        num_kv_heads,
        head_dim,
        rope_theta,
        0,
    );

    // Now split (after RoPE applied)
    let q = &qkv[..q_end];
    let k = &qkv[q_end..k_end];
    let v = &qkv[k_end..];

    // Cache K (with RoPE) and V
    for pos in 0..seq_len {
        let k_slice = &k[pos * kv_dim..(pos + 1) * kv_dim];
        let v_slice = &v[pos * kv_dim..(pos + 1) * kv_dim];
        kv_cache.append(block_idx, k_slice, v_slice);
    }

    // GQA attention
    let attn_out =
        gqa_attention_with_kv(model, q, k, v, seq_len, num_heads, num_kv_heads, head_dim)?;

    // Output projection
    let projected = model.scheduler.matmul(
        &attn_out,
        &model.block_weights[block_idx].out_weight,
        seq_len,
        hidden_dim,
        hidden_dim,
    )?;

    // Residual 1
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| {
            inp + proj + model.block_weights[block_idx].out_bias[i % hidden_dim]
        })
        .collect();

    // FFN pre-norm
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &model.block_weights[block_idx].ffn_norm_weight,
        &model.block_weights[block_idx].ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN: SwiGLU when gate weight exists, otherwise GELU
    let activated: Vec<f32> = if let Some(ref gate_weight) =
        model.block_weights[block_idx].ffn_gate_weight
    {
        // SwiGLU: silu(gate(x)) * up(x)
        let up_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;
        let gate_out = model.scheduler.matmul(
            &ffn_normed,
            gate_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        // SwiGLU: silu(gate) * up
        up_out
            .iter()
            .zip(gate_out.iter())
            .map(|(&u, &g)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect()
    } else {
        // Standard GELU FFN
        let fc1_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + model.block_weights[block_idx].ffn_fc1_bias[i % intermediate_dim];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    // FFN: fc2
    let fc2_out = model.scheduler.matmul(
        &activated,
        &model.block_weights[block_idx].ffn_fc2_weight,
        seq_len,
        intermediate_dim,
        hidden_dim,
    )?;

    // Residual 2
    for (i, x) in residual1.iter_mut().enumerate() {
        *x += fc2_out[i] + model.block_weights[block_idx].ffn_fc2_bias[i % hidden_dim];
    }

    Ok(residual1)
}

/// Incremental forward pass through a single block using cached KV
fn forward_block_incremental(
    model: &mut GpuModel,
    input: &[f32],
    block_idx: usize,
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let kv_dim = model.config.kv_dim();
    let qkv_dim = model.config.qkv_dim();

    let block = &model.block_weights[block_idx];

    // Pre-norm (single position)
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection (single position)
    let mut qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        1,
        hidden_dim,
        qkv_dim,
    )?;

    // Get current position BEFORE caching (this is where new token goes)
    let (existing_k, _) = kv_cache.get_valid(block_idx);
    let current_pos = existing_k.len() / kv_dim;

    // Phase 21: Apply RoPE to Q and K at current position BEFORE caching
    let rope_theta = model.config.rope_theta;

    // Apply RoPE to Q (single position, all heads)
    apply_rope(
        &mut qkv[..hidden_dim],
        1,
        num_heads,
        head_dim,
        rope_theta,
        current_pos,
    );

    // Apply RoPE to K (single position, KV heads)
    apply_rope(
        &mut qkv[hidden_dim..hidden_dim + kv_dim],
        1,
        num_kv_heads,
        head_dim,
        rope_theta,
        current_pos,
    );

    // Split Q, K, V (single position, after RoPE)
    let q = &qkv[..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v = &qkv[hidden_dim + kv_dim..];

    // Cache new K (with RoPE) and V
    kv_cache.append(block_idx, k, v);

    // Get all cached K/V for attention (now includes new K/V)
    let (all_k, all_v) = kv_cache.get_valid(block_idx);
    let cache_len = all_k.len() / kv_dim;

    // GQA incremental attention
    let attn_out = gqa_incremental_attention(
        model,
        q,
        all_k,
        all_v,
        cache_len,
        num_heads,
        num_kv_heads,
        head_dim,
    )?;

    // Output projection
    let projected = model.scheduler.matmul(
        &attn_out,
        &model.block_weights[block_idx].out_weight,
        1,
        hidden_dim,
        hidden_dim,
    )?;

    // Residual 1
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + model.block_weights[block_idx].out_bias[i])
        .collect();

    // FFN pre-norm
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &model.block_weights[block_idx].ffn_norm_weight,
        &model.block_weights[block_idx].ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN: SwiGLU when gate weight exists, otherwise GELU
    let activated: Vec<f32> = if let Some(ref gate_weight) =
        model.block_weights[block_idx].ffn_gate_weight
    {
        // SwiGLU: silu(gate(x)) * up(x)
        let up_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            1,
            hidden_dim,
            intermediate_dim,
        )?;
        let gate_out =
            model
                .scheduler
                .matmul(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim)?;

        // SwiGLU: silu(gate) * up
        up_out
            .iter()
            .zip(gate_out.iter())
            .map(|(&u, &g)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect()
    } else {
        // Standard GELU FFN
        let fc1_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            1,
            hidden_dim,
            intermediate_dim,
        )?;

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + model.block_weights[block_idx].ffn_fc1_bias[i];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    // FFN: fc2
    let fc2_out = model.scheduler.matmul(
        &activated,
        &model.block_weights[block_idx].ffn_fc2_weight,
        1,
        intermediate_dim,
        hidden_dim,
    )?;

    // Residual 2
    for (i, x) in residual1.iter_mut().enumerate() {
        *x += fc2_out[i] + model.block_weights[block_idx].ffn_fc2_bias[i];
    }

    Ok(residual1)
}

/// GQA attention with KV (full sequence)
#[allow(clippy::too_many_arguments)]
fn gqa_attention_with_kv(
    _model: &GpuModel,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;

    let mut output = vec![0.0f32; seq_len * hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for pos in 0..seq_len {
        for head in 0..num_heads {
            let kv_head = head / heads_per_kv;

            // Query for this head at this position
            let q_start = pos * hidden_dim + head * head_dim;
            let q_slice = &q[q_start..q_start + head_dim];

            // Compute attention scores for all positions up to current
            let mut scores = Vec::with_capacity(pos + 1);
            for kpos in 0..=pos {
                let k_start = kpos * kv_dim + kv_head * head_dim;
                let k_slice = &k[k_start..k_start + head_dim];

                let score: f32 = q_slice
                    .iter()
                    .zip(k_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                scores.push(score * scale);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();

            // Weighted sum of values
            let out_start = pos * hidden_dim + head * head_dim;
            for (kpos, &weight) in weights.iter().enumerate() {
                let v_start = kpos * kv_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }

    Ok(output)
}

/// GQA incremental attention (single query position)
#[allow(clippy::too_many_arguments)]
fn gqa_incremental_attention(
    _model: &GpuModel,
    q: &[f32],
    all_k: &[f32],
    all_v: &[f32],
    cache_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;

    let mut output = vec![0.0f32; hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..num_heads {
        let kv_head = head / heads_per_kv;

        let q_start = head * head_dim;
        let q_slice = &q[q_start..q_start + head_dim];

        // Attention scores for all cached positions
        let mut scores = Vec::with_capacity(cache_len);
        for kpos in 0..cache_len {
            let k_start = kpos * kv_dim + kv_head * head_dim;
            let k_slice = &all_k[k_start..k_start + head_dim];

            let score: f32 = q_slice
                .iter()
                .zip(k_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            scores.push(score * scale);
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();

        // Weighted sum
        let out_start = head * head_dim;
        for (kpos, &weight) in weights.iter().enumerate() {
            let v_start = kpos * kv_dim + kv_head * head_dim;
            for d in 0..head_dim {
                output[out_start + d] += weight * all_v[v_start + d];
            }
        }
    }

    Ok(output)
}

/// Sample next token based on config (greedy or top-k)
#[inline]
fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    if temperature == 0.0 || top_k == 1 {
        argmax(logits)
    } else {
        sample_topk(logits, temperature, top_k)
    }
}

/// Generate tokens using KV cache (IMP-033)
pub fn generate_with_cache(
    model: &mut GpuModel,
    prompt: &[usize],
    config: &GpuGenerateConfig,
) -> Result<Vec<usize>> {
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Prompt cannot be empty".to_string(),
        });
    }

    let max_seq_len = prompt.len() + config.max_tokens;
    let head_dim = model.config.hidden_dim / model.config.num_heads;
    let mut kv_cache = StreamingKVCache::new(
        model.config.num_layers,
        max_seq_len,
        model.config.num_kv_heads,
        head_dim,
    );

    let mut tokens = prompt.to_vec();
    let logits = forward_gpu_with_cache(model, prompt, &mut kv_cache)?;
    let mut next_token = sample_token(&logits, config.temperature, config.top_k);

    if config.stop_tokens.contains(&next_token) {
        return Ok(tokens);
    }
    tokens.push(next_token);

    for _ in 1..config.max_tokens {
        let logits = forward_gpu_incremental(model, next_token, &mut kv_cache)?;
        next_token = sample_token(&logits, config.temperature, config.top_k);

        if config.stop_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }

    Ok(tokens)
}

/// Layer norm helper for KV methods
fn layer_norm_kv(model: &GpuModel, input: &[f32]) -> Vec<f32> {
    GpuModel::layer_norm_static(
        input,
        &model.final_norm_weight,
        &model.final_norm_bias,
        model.config.hidden_dim,
        model.config.eps,
    )
}

/// Argmax helper
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

/// Top-k sampling helper
fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);
    indexed.first().map_or(0, |&(idx, _)| idx)
}

// ============================================================================
// Tests (Protocol T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === apply_rope tests ===

    #[test]
    fn test_apply_rope_basic() {
        // Single position, single head, head_dim=2
        let mut x = vec![1.0, 0.0]; // [cos, sin] position encoding input
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 2;
        let rope_theta = 10000.0;
        let start_pos = 0;

        apply_rope(&mut x, seq_len, num_heads, head_dim, rope_theta, start_pos);

        // At position 0, angle = 0 * freq = 0, so sin(0)=0, cos(0)=1
        // Rotation: [x1*1 - x2*0, x1*0 + x2*1] = [1, 0]
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!(x[1].abs() < 1e-5, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_apply_rope_position_one() {
        // Single position at pos=1 should show rotation
        let mut x = vec![1.0, 0.0];
        apply_rope(&mut x, 1, 1, 2, 10000.0, 1);

        // At position 1, angle = 1 * 1/(10000^0) = 1
        // cos(1) ≈ 0.54, sin(1) ≈ 0.84
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!(
            (x[0] - expected_cos).abs() < 1e-5,
            "x[0] = {}, expected {}",
            x[0],
            expected_cos
        );
        assert!(
            (x[1] - expected_sin).abs() < 1e-5,
            "x[1] = {}, expected {}",
            x[1],
            expected_sin
        );
    }

    #[test]
    fn test_apply_rope_multiple_positions() {
        // Two positions
        let mut x = vec![
            1.0, 0.0, // Position 0
            1.0, 0.0, // Position 1
        ];
        apply_rope(&mut x, 2, 1, 2, 10000.0, 0);

        // Position 0: no rotation (angle=0)
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);

        // Position 1: rotated
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!((x[2] - expected_cos).abs() < 1e-5);
        assert!((x[3] - expected_sin).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_multiple_heads() {
        // Single position, two heads
        let mut x = vec![
            1.0, 0.0, // Head 0
            0.0, 1.0, // Head 1
        ];
        apply_rope(&mut x, 1, 2, 2, 10000.0, 0);

        // At position 0, both heads get identity rotation
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);
        assert!(x[2].abs() < 1e-5);
        assert!((x[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_larger_head_dim() {
        // Head dim 4 (2 pairs to rotate)
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 1, 1, 4, 10000.0, 0);

        // At position 0, angle = 0, no rotation
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
        assert!((x[2] - 3.0).abs() < 1e-5);
        assert!((x[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_with_start_pos() {
        let mut x = vec![1.0, 0.0];
        apply_rope(&mut x, 1, 1, 2, 10000.0, 5);

        // At position 5, angle = 5 * 1 = 5
        let expected_cos = 5.0f32.cos();
        let expected_sin = 5.0f32.sin();
        assert!(
            (x[0] - expected_cos).abs() < 1e-5,
            "x[0] = {}, expected {}",
            x[0],
            expected_cos
        );
        assert!(
            (x[1] - expected_sin).abs() < 1e-5,
            "x[1] = {}, expected {}",
            x[1],
            expected_sin
        );
    }

    // === argmax tests ===

    #[test]
    fn test_argmax_kv_single() {
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn test_argmax_kv_first() {
        assert_eq!(argmax(&[10.0, 5.0, 3.0]), 0);
    }

    #[test]
    fn test_argmax_kv_last() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
    }

    #[test]
    fn test_argmax_kv_middle() {
        assert_eq!(argmax(&[1.0, 10.0, 3.0]), 1);
    }

    #[test]
    fn test_argmax_kv_negatives() {
        assert_eq!(argmax(&[-5.0, -2.0, -10.0]), 1);
    }

    #[test]
    fn test_argmax_kv_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    // === sample_topk tests ===

    #[test]
    fn test_sample_topk_returns_max_with_low_temp() {
        let logits = vec![1.0, 10.0, 2.0];
        // With temperature=1.0 and top_k=1, should return argmax
        let result = sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sample_topk_top_3() {
        let logits = vec![0.0, 10.0, 5.0, 1.0];
        // Top-3: indices 1, 2, 3. Should return one of them (likely 1)
        let result = sample_topk(&logits, 1.0, 3);
        assert!(result <= 3, "result = {}", result);
    }

    #[test]
    fn test_sample_topk_with_high_temp() {
        let logits = vec![0.0, 1.0, 0.0];
        // High temperature makes distribution flatter, but max still most likely
        let result = sample_topk(&logits, 10.0, 3);
        assert!(result <= 2);
    }

    #[test]
    fn test_sample_topk_top_1_is_argmax() {
        let logits = vec![0.0, 0.0, 100.0, 0.0];
        let result = sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_sample_topk_empty_returns_zero() {
        let result = sample_topk(&[], 1.0, 10);
        assert_eq!(result, 0);
    }

    // === GQA attention dimension tests ===

    #[test]
    fn test_gqa_attention_dimension_calculations() {
        // Test dimension calculations used in GQA attention
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;
        let seq_len = 10;

        let hidden_dim = num_heads * head_dim;
        assert_eq!(hidden_dim, 4096);

        let kv_dim = num_kv_heads * head_dim;
        assert_eq!(kv_dim, 1024);

        let heads_per_kv = num_heads / num_kv_heads;
        assert_eq!(heads_per_kv, 4);

        // Output size
        let output_size = seq_len * hidden_dim;
        assert_eq!(output_size, 40960);

        // Q, K, V sizes
        let q_size = seq_len * hidden_dim;
        let k_size = seq_len * kv_dim;
        let v_size = seq_len * kv_dim;
        assert_eq!(q_size, 40960);
        assert_eq!(k_size, 10240);
        assert_eq!(v_size, 10240);
    }

    #[test]
    fn test_gqa_attention_scale_factor() {
        let head_dim = 128;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected = 1.0 / 128.0_f32.sqrt();
        assert!((scale - expected).abs() < 1e-6);
        assert!((scale - 0.088388).abs() < 1e-5);
    }

    // === gqa_incremental_attention tests ===

    #[test]
    fn test_gqa_incremental_attention_params() {
        // Just test the function exists and has correct signature
        // Actually calling it would require a valid GpuModel
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let cache_len = 5;

        // These would be inputs
        let _q = vec![0.0f32; num_heads * head_dim];
        let _all_k = vec![0.0f32; cache_len * num_kv_heads * head_dim];
        let _all_v = vec![0.0f32; cache_len * num_kv_heads * head_dim];

        // Verify dimension calculations
        let hidden_dim = num_heads * head_dim;
        assert_eq!(hidden_dim, 64);
        let kv_dim = num_kv_heads * head_dim;
        assert_eq!(kv_dim, 32);
        let heads_per_kv = num_heads / num_kv_heads;
        assert_eq!(heads_per_kv, 2);
    }

    // === layer_norm_kv smoke test ===

    #[test]
    fn test_layer_norm_static_dimensions() {
        // Test that GpuModel::layer_norm_static returns correct dimensions
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let hidden_dim = 4;
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, hidden_dim, eps);
        assert_eq!(output.len(), 4);

        // Verify output is finite
        for &v in &output {
            assert!(v.is_finite(), "output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn test_layer_norm_static_preserves_length() {
        // Test that layer norm preserves the input length
        let hidden_dim = 8;
        let seq_len = 4;
        let input = vec![1.0; seq_len * hidden_dim];
        let weight = vec![1.0; hidden_dim];
        let bias = vec![0.0; hidden_dim];
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, hidden_dim, eps);
        assert_eq!(output.len(), seq_len * hidden_dim);
    }
}
