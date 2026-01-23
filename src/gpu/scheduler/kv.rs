//! KV Cache Management for GpuModel (PMAT-802)
//!
//! Extracted from model.rs to reduce module size.
//! Contains KV cache forward pass and generation logic.

use crate::error::{RealizarError, Result};
use super::super::{StreamingKVCache, exceeds_gpu_buffer_limit, cpu_matmul_transposed_simd};
use super::model::{GpuModel, GpuGenerateConfig};

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
    let qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        seq_len,
        hidden_dim,
        qkv_dim,
    )?;

    // Split Q, K, V
    let q = &qkv[..seq_len * hidden_dim];
    let k = &qkv[seq_len * hidden_dim..seq_len * hidden_dim + seq_len * kv_dim];
    let v = &qkv[seq_len * hidden_dim + seq_len * kv_dim..];

    // Cache K and V
    for pos in 0..seq_len {
        let k_slice = &k[pos * kv_dim..(pos + 1) * kv_dim];
        let v_slice = &v[pos * kv_dim..(pos + 1) * kv_dim];
        kv_cache.append(block_idx, k_slice, v_slice);
    }

    // GQA attention
    let attn_out = gqa_attention_with_kv(model, q, k, v, seq_len, num_heads, num_kv_heads, head_dim)?;

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

    // FFN: fc1
    let fc1_out = model.scheduler.matmul(
        &ffn_normed,
        &model.block_weights[block_idx].ffn_fc1_weight,
        seq_len,
        hidden_dim,
        intermediate_dim,
    )?;

    // GELU activation + bias
    let activated: Vec<f32> = fc1_out
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let x = x + model.block_weights[block_idx].ffn_fc1_bias[i % intermediate_dim];
            0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
        })
        .collect();

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
    let qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        1,
        hidden_dim,
        qkv_dim,
    )?;

    // Split Q, K, V (single position)
    let q = &qkv[..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v = &qkv[hidden_dim + kv_dim..];

    // Cache new K/V
    kv_cache.append(block_idx, k, v);

    // Get all cached K/V for attention
    let (all_k, all_v) = kv_cache.get_valid(block_idx);
    let cache_len = all_k.len() / kv_dim;

    // GQA incremental attention
    let attn_out = gqa_incremental_attention(model, q, all_k, all_v, cache_len, num_heads, num_kv_heads, head_dim)?;

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
        .map(|(i, (&inp, &proj))| {
            inp + proj + model.block_weights[block_idx].out_bias[i]
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

    // FFN: fc1
    let fc1_out = model.scheduler.matmul(
        &ffn_normed,
        &model.block_weights[block_idx].ffn_fc1_weight,
        1,
        hidden_dim,
        intermediate_dim,
    )?;

    // GELU + bias
    let activated: Vec<f32> = fc1_out
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let x = x + model.block_weights[block_idx].ffn_fc1_bias[i];
            0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
        })
        .collect();

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

                let score: f32 = q_slice.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
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

            let score: f32 = q_slice.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
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

    let mut next_token = if config.temperature == 0.0 || config.top_k == 1 {
        argmax(&logits)
    } else {
        sample_topk(&logits, config.temperature, config.top_k)
    };

    if config.stop_tokens.contains(&next_token) {
        return Ok(tokens);
    }
    tokens.push(next_token);

    for _ in 1..config.max_tokens {
        let logits = forward_gpu_incremental(model, next_token, &mut kv_cache)?;

        next_token = if config.temperature == 0.0 || config.top_k == 1 {
            argmax(&logits)
        } else {
            sample_topk(&logits, config.temperature, config.top_k)
        };

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
