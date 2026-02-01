//! Batch Generation and Single-Token Forward (PMAT-802)
//!
//! Extracted from model.rs: incremental generation, single-token forward, and helpers.

use super::super::{cpu_matmul, cpu_matmul_transposed_simd, exceeds_gpu_buffer_limit};
use super::model::GpuModel;
use super::types::GpuModelConfig;
use crate::error::{RealizarError, Result};

/// Generate tokens using GPU-accelerated forward pass with incremental decoding
///
/// # Arguments
///
/// * `model` - GPU model reference
/// * `prompt` - Initial token IDs
/// * `max_tokens` - Maximum tokens to generate
///
/// # Returns
///
/// Generated tokens (including prompt)
///
/// # Errors
///
/// Returns error if generation fails
pub fn generate_gpu(
    model: &mut GpuModel,
    prompt: &[usize],
    max_tokens: usize,
) -> Result<Vec<usize>> {
    let mut tokens = prompt.to_vec();
    let vocab_size = model.config.vocab_size;

    // Process prompt first (full forward)
    let logits = model.forward_gpu(&tokens)?;

    // Get first prediction
    let last_pos_start = (tokens.len() - 1) * vocab_size;
    let last_logits = &logits[last_pos_start..last_pos_start + vocab_size];

    let next_token = argmax(last_logits);
    tokens.push(next_token);

    // Generate remaining tokens one at a time (incremental)
    // Use optimized greedy path for large vocabularies
    if vocab_size > 8192 {
        // Large vocab: use fused LM head + argmax
        for _ in 1..max_tokens {
            let next_token = forward_single_token_greedy(model, &tokens)?;
            tokens.push(next_token);
        }
    } else {
        // Small vocab: standard path
        for _ in 1..max_tokens {
            let logits = forward_single_token(model, &tokens)?;
            let next_token = argmax(&logits);
            tokens.push(next_token);
        }
    }

    Ok(tokens)
}

/// Fast single-token forward pass for incremental generation
///
/// Only processes the last token position, avoiding O(n²) recomputation.
pub fn forward_single_token(model: &mut GpuModel, tokens: &[usize]) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let vocab_size = model.config.vocab_size;

    // Embed only the last token
    let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
        reason: "Token list empty".to_string(),
    })?;

    if last_token >= vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!("Token {} out of bounds", last_token),
        });
    }

    let offset = last_token * hidden_dim;
    let mut hidden: Vec<f32> = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Process through blocks (simplified for single token)
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_single(model, &hidden, block_idx)?;
    }

    // Final layer norm
    hidden = GpuModel::layer_norm_static(
        &hidden,
        &model.final_norm_weight,
        &model.final_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // IMP-090, IMP-096: Use CPU fallback with SIMD for large vocab
    let lm_head_elements = hidden_dim * vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        // IMP-096: CPU path with transposed weights + SIMD + fused bias
        // Uses parallel dot products with perfect cache behavior
        cpu_matmul_transposed_simd(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            vocab_size,
        )
    } else {
        // GPU path for smaller vocab
        // Phase 44: Use do_matmul() to enable MockExecutor testing
        let lm_head_weight = model.lm_head_weight.clone();
        let logits = model.do_matmul(&hidden, &lm_head_weight, 1, hidden_dim, vocab_size)?;
        // Add bias
        logits
            .iter()
            .zip(model.lm_head_bias.iter())
            .map(|(&x, &b)| x + b)
            .collect()
    };

    Ok(output)
}

/// Single-token forward pass optimized for greedy sampling
///
/// Returns the argmax token directly.
pub fn forward_single_token_greedy(model: &mut GpuModel, tokens: &[usize]) -> Result<usize> {
    let hidden_dim = model.config.hidden_dim;
    let vocab_size = model.config.vocab_size;

    // Embed only the last token
    let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
        reason: "Token list empty".to_string(),
    })?;

    if last_token >= vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!("Token {} out of bounds", last_token),
        });
    }

    let offset = last_token * hidden_dim;
    let mut hidden: Vec<f32> = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Process through blocks (simplified for single token)
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_single(model, &hidden, block_idx)?;
    }

    // Final layer norm
    hidden = GpuModel::layer_norm_static(
        &hidden,
        &model.final_norm_weight,
        &model.final_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // Use optimized CPU path with transposed weights for large vocab
    // This uses row-major access pattern which is ~3-5x faster than column access
    // IMP-090: Also use CPU path if vocab would exceed GPU buffer limits
    let lm_head_elements = hidden_dim * vocab_size;
    if vocab_size > 8192 || exceeds_gpu_buffer_limit(lm_head_elements) {
        // CPU path with transposed weights: perfect cache behavior
        Ok(optimized_lm_head_argmax_transposed(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            vocab_size,
        ))
    } else {
        // GPU/small vocab path
        // Phase 44: Use do_matmul() to enable MockExecutor testing
        let lm_head_weight = model.lm_head_weight.clone();
        let logits = model.do_matmul(&hidden, &lm_head_weight, 1, hidden_dim, vocab_size)?;
        let output: Vec<f32> = logits
            .iter()
            .zip(model.lm_head_bias.iter())
            .map(|(&x, &b)| x + b)
            .collect();
        Ok(argmax(&output))
    }
}

/// Single token forward through a transformer block (CPU-optimized for m=1)
///
/// For single-token generation, CPU operations are faster than GPU due to transfer overhead.
#[allow(clippy::unnecessary_wraps)]
pub fn forward_block_single(
    model: &mut GpuModel,
    input: &[f32],
    block_idx: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let kv_dim = model.config.kv_dim();
    let qkv_dim = model.config.qkv_dim();

    // Get block weights
    let block = &model.block_weights[block_idx];

    // Pre-norm
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection for single token (GQA: qkv_dim = hidden_dim + 2*kv_dim)
    // Use CPU matmul directly - GPU overhead not worth it for m=1
    let qkv_weight = &model.block_weights[block_idx].qkv_weight;
    let qkv = cpu_matmul(&normed, qkv_weight, 1, hidden_dim, qkv_dim);

    // Split QKV and apply simplified self-attention (single token)
    // q and k unused for single-token (no cross-attention needed)
    // GQA: V has kv_dim size, but we need hidden_dim output
    let v = &qkv[hidden_dim + kv_dim..];

    // For single token: attention output = v (self-attention with one token)
    // GQA: V has kv_dim, need to repeat heads to get hidden_dim
    let num_kv_heads = model.config.num_kv_heads;
    let heads_per_kv = model.config.num_heads / num_kv_heads;
    let head_dim = model.config.head_dim();

    let attn_out: Vec<f32> = if heads_per_kv == 1 {
        // Standard MHA: no repetition needed
        v.to_vec()
    } else {
        // GQA: repeat each KV head to serve multiple Q heads
        let mut expanded = Vec::with_capacity(hidden_dim);
        for kv_h in 0..num_kv_heads {
            let v_head = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
            for _ in 0..heads_per_kv {
                expanded.extend_from_slice(v_head);
            }
        }
        expanded
    };

    // Output projection (CPU - m=1)
    let out_weight = &model.block_weights[block_idx].out_weight;
    let out_bias = &model.block_weights[block_idx].out_bias;
    let projected = cpu_matmul(&attn_out, out_weight, 1, hidden_dim, hidden_dim);

    // Residual 1
    let residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + out_bias[i])
        .collect();

    // FFN pre-norm
    let ffn_norm_weight = &model.block_weights[block_idx].ffn_norm_weight;
    let ffn_norm_bias = &model.block_weights[block_idx].ffn_norm_bias;
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        ffn_norm_weight,
        ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN fc1 (CPU - m=1)
    let ffn_fc1_weight = &model.block_weights[block_idx].ffn_fc1_weight;
    let ffn_fc1_bias = &model.block_weights[block_idx].ffn_fc1_bias;

    // FFN: SwiGLU when gate weight exists, otherwise GELU
    let activated: Vec<f32> = if let Some(ref gate_weight) =
        model.block_weights[block_idx].ffn_gate_weight
    {
        // SwiGLU: silu(gate(x)) * up(x)
        let up_out = cpu_matmul(&ffn_normed, ffn_fc1_weight, 1, hidden_dim, intermediate_dim);
        let gate_out = cpu_matmul(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim);

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
        let fc1_out = cpu_matmul(&ffn_normed, ffn_fc1_weight, 1, hidden_dim, intermediate_dim);

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + ffn_fc1_bias[i];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    // FFN fc2 (CPU - m=1)
    let ffn_fc2_weight = &model.block_weights[block_idx].ffn_fc2_weight;
    let ffn_fc2_bias = &model.block_weights[block_idx].ffn_fc2_bias;
    let fc2_out = cpu_matmul(&activated, ffn_fc2_weight, 1, intermediate_dim, hidden_dim);

    // Residual 2
    let output: Vec<f32> = residual1
        .iter()
        .zip(fc2_out.iter())
        .enumerate()
        .map(|(i, (&r, &fc))| r + fc + ffn_fc2_bias[i])
        .collect();

    Ok(output)
}

/// Argmax helper for sampling - vectorized for large vocabularies
#[allow(clippy::items_after_statements)]
pub fn argmax(logits: &[f32]) -> usize {
    // For small vocab, use simple iterator
    if logits.len() <= 1024 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
    }

    // For large vocab (32K+), use chunked parallel argmax
    const CHUNK_SIZE: usize = 4096;

    // Find max in each chunk
    let chunk_maxes: Vec<(usize, f32)> = logits
        .chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let (local_idx, &max_val) = chunk
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("chunk is non-empty by construction");
            (chunk_idx * CHUNK_SIZE + local_idx, max_val)
        })
        .collect();

    // Find global max
    chunk_maxes
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

/// Optimized LM head + argmax using transposed weights with vectorized dot products
///
/// Uses transposed weights [vocab_size, hidden_dim] for row-major access pattern.
/// Inner loop is vectorized by the compiler via slice operations.
#[allow(clippy::many_single_char_names, clippy::items_after_statements)]
pub fn optimized_lm_head_argmax_transposed(
    hidden: &[f32],
    weight_t: &[f32], // Transposed: [vocab_size, hidden_dim]
    bias: &[f32],
    hidden_dim: usize,
    vocab_size: usize,
) -> usize {
    use rayon::prelude::*;

    // Process in larger chunks for better parallelism
    const CHUNK_SIZE: usize = 4096;

    // Find argmax in parallel
    (0..vocab_size)
        .into_par_iter()
        .step_by(CHUNK_SIZE)
        .map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(vocab_size);
            let mut best_local_idx = chunk_start;
            let mut best_local_val = f32::NEG_INFINITY;

            for j in chunk_start..chunk_end {
                // Row-major access: weight_t[j, :] is contiguous in memory
                let row = &weight_t[j * hidden_dim..(j + 1) * hidden_dim];

                // Vectorized dot product - compiler can auto-vectorize this
                let dot: f32 = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();

                let logit = dot + bias[j];

                if logit > best_local_val {
                    best_local_val = logit;
                    best_local_idx = j;
                }
            }
            (best_local_idx, best_local_val)
        })
        .reduce(
            || (0, f32::NEG_INFINITY),
            |a, b| if a.1 > b.1 { a } else { b },
        )
        .0
}

/// Extract Q tensor for a single head from packed Q data
fn extract_q_head(q: &[f32], head: usize, seq_len: usize, hidden_dim: usize, head_dim: usize) -> Vec<f32> {
    let mut q_head = Vec::with_capacity(seq_len * head_dim);
    for i in 0..seq_len {
        let start = i * hidden_dim + head * head_dim;
        q_head.extend_from_slice(&q[start..start + head_dim]);
    }
    q_head
}

/// Extract K and V tensors for a KV head from packed K/V data
fn extract_kv_head(k: &[f32], v: &[f32], kv_head: usize, seq_len: usize, kv_dim: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k_head = Vec::with_capacity(seq_len * head_dim);
    let mut v_head = Vec::with_capacity(seq_len * head_dim);
    for i in 0..seq_len {
        let start = i * kv_dim + kv_head * head_dim;
        k_head.extend_from_slice(&k[start..start + head_dim]);
        v_head.extend_from_slice(&v[start..start + head_dim]);
    }
    (k_head, v_head)
}

/// Apply causal softmax to attention scores
fn apply_causal_softmax(scores: &[f32], seq_len: usize, scale: f32) -> Vec<f32> {
    let mut attn = vec![f32::NEG_INFINITY; seq_len * seq_len];

    // Apply causal mask and scale
    for i in 0..seq_len {
        for j in 0..=i {
            attn[i * seq_len + j] = scores[i * seq_len + j] * scale;
        }
    }

    // Softmax per row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row = &mut attn[row_start..row_start + seq_len];
        let max_val = row[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for item in row.iter_mut().take(i + 1) {
            *item = (*item - max_val).exp();
            sum += *item;
        }
        for item in row.iter_mut().take(i + 1) {
            *item /= sum;
        }
        for item in row.iter_mut().skip(i + 1) {
            *item = 0.0;
        }
    }
    attn
}

/// Optimized GQA attention using GPU for matmul operations (IMP-089)
pub fn optimized_gqa_attention(
    model: &mut GpuModel,
    qkv: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let kv_dim = model.config.kv_dim();
    let heads_per_kv = num_heads / num_kv_heads;

    // Split QKV (GQA: K/V have kv_dim per position)
    let q = &qkv[..seq_len * hidden_dim];
    let k = &qkv[seq_len * hidden_dim..seq_len * hidden_dim + seq_len * kv_dim];
    let v = &qkv[seq_len * hidden_dim + seq_len * kv_dim..];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..num_heads {
        let kv_head = head / heads_per_kv;
        let q_head = extract_q_head(q, head, seq_len, hidden_dim, head_dim);
        let (k_head, v_head) = extract_kv_head(k, v, kv_head, seq_len, kv_dim, head_dim);

        // Compute attention scores: Q @ K^T using GPU matmul
        let scores = model.do_matmul_transpose_b(&q_head, &k_head, seq_len, head_dim, seq_len)?;
        let attn_scores = apply_causal_softmax(&scores, seq_len, scale);

        // Compute output: attn @ V using GPU matmul
        let head_output = model.do_matmul(&attn_scores, &v_head, seq_len, seq_len, head_dim)?;

        // Copy to output
        for i in 0..seq_len {
            let out_start = i * hidden_dim + head * head_dim;
            let head_start = i * head_dim;
            output[out_start..out_start + head_dim]
                .copy_from_slice(&head_output[head_start..head_start + head_dim]);
        }
    }

    Ok(output)
}

/// Simplified attention (fallback, for M3 benchmarking)
#[allow(dead_code, clippy::unnecessary_wraps)]
pub fn simplified_attention(
    config: &GpuModelConfig,
    qkv: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads;

    // Split QKV
    let q = &qkv[..seq_len * hidden_dim];
    let k = &qkv[seq_len * hidden_dim..seq_len * 2 * hidden_dim];
    let v = &qkv[seq_len * 2 * hidden_dim..];

    // Simplified scaled dot-product attention per head
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..config.num_heads {
        for i in 0..seq_len {
            // Compute attention weights for position i
            let mut weights = Vec::with_capacity(seq_len);
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..=i {
                // Causal: only attend to previous positions
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = i * hidden_dim + head * head_dim + d;
                    let k_idx = j * hidden_dim + head * head_dim + d;
                    score += q[q_idx] * k[k_idx];
                }
                score *= scale;
                max_score = max_score.max(score);
                weights.push(score);
            }

            // Softmax
            let mut sum = 0.0f32;
            for w in &mut weights {
                *w = (*w - max_score).exp();
                sum += *w;
            }
            for w in &mut weights {
                *w /= sum;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let out_idx = i * hidden_dim + head * head_dim + d;
                for (j, &w) in weights.iter().enumerate() {
                    let v_idx = j * hidden_dim + head * head_dim + d;
                    output[out_idx] += w * v[v_idx];
                }
            }
        }
    }

    Ok(output)
}

// ============================================================================
// Tests (Protocol T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === argmax tests ===

    #[test]
    fn test_argmax_single_element() {
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn test_argmax_first_is_max() {
        assert_eq!(argmax(&[10.0, 5.0, 3.0, 1.0]), 0);
    }

    #[test]
    fn test_argmax_last_is_max() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 4.0]), 3);
    }

    #[test]
    fn test_argmax_middle_is_max() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn test_argmax_with_negatives() {
        assert_eq!(argmax(&[-5.0, -1.0, -10.0]), 1);
    }

    #[test]
    fn test_argmax_with_equal_values() {
        // First max wins
        let result = argmax(&[3.0, 3.0, 3.0]);
        assert!(result <= 2);
    }

    #[test]
    fn test_argmax_large_vocab_uses_chunked_path() {
        // Create logits larger than 1024 to trigger chunked path
        let mut logits = vec![0.0f32; 2000];
        logits[1500] = 10.0;
        assert_eq!(argmax(&logits), 1500);
    }

    #[test]
    fn test_argmax_large_vocab_first_chunk() {
        let mut logits = vec![0.0f32; 5000];
        logits[100] = 10.0;
        assert_eq!(argmax(&logits), 100);
    }

    #[test]
    fn test_argmax_large_vocab_last_chunk() {
        let mut logits = vec![0.0f32; 5000];
        logits[4999] = 10.0;
        assert_eq!(argmax(&logits), 4999);
    }

    #[test]
    fn test_argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    // === optimized_lm_head_argmax_transposed tests ===

    #[test]
    fn test_lm_head_argmax_basic() {
        // 4 vocab, 2 hidden
        let hidden = vec![1.0, 2.0];
        // Transposed weights: each row is weights for one vocab token
        // Token 0: [0.0, 0.0] -> dot = 0
        // Token 1: [1.0, 0.0] -> dot = 1
        // Token 2: [0.0, 1.0] -> dot = 2
        // Token 3: [1.0, 1.0] -> dot = 3
        let weight_t = vec![
            0.0, 0.0, // Token 0
            1.0, 0.0, // Token 1
            0.0, 1.0, // Token 2
            1.0, 1.0, // Token 3
        ];
        let bias = vec![0.0; 4];

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 4);
        assert_eq!(result, 3); // Token 3 has highest dot product
    }

    #[test]
    fn test_lm_head_argmax_with_bias() {
        let hidden = vec![1.0, 1.0];
        let weight_t = vec![
            1.0, 1.0, // Token 0: dot = 2
            0.0, 0.0, // Token 1: dot = 0
        ];
        let bias = vec![0.0, 10.0]; // Token 1 gets big bias

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 2);
        assert_eq!(result, 1); // Bias makes token 1 win
    }

    #[test]
    fn test_lm_head_argmax_negative_weights() {
        let hidden = vec![1.0, 1.0];
        let weight_t = vec![
            -1.0, -1.0, // Token 0: dot = -2
            1.0, 1.0, // Token 1: dot = 2
        ];
        let bias = vec![0.0, 0.0];

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 2);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_lm_head_argmax_large_vocab() {
        // Test with larger vocabulary to exercise parallel chunks
        let hidden_dim = 64;
        let vocab_size = 10000;
        let hidden = vec![1.0; hidden_dim];

        // Set up weights so token 5000 wins
        let mut weight_t = vec![0.0; vocab_size * hidden_dim];
        for i in 0..hidden_dim {
            weight_t[5000 * hidden_dim + i] = 1.0;
        }
        let bias = vec![0.0; vocab_size];

        let result =
            optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
        assert_eq!(result, 5000);
    }

    // === simplified_attention tests ===

    #[test]
    fn test_simplified_attention_single_position() {
        let config = GpuModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 8,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // Single position: QKV for seq_len=1
        // Q: [1, 0, 1, 0], K: [1, 0, 1, 0], V: [0.5, 0.5, 0.5, 0.5]
        let qkv = vec![
            1.0, 0.0, 1.0, 0.0, // Q
            1.0, 0.0, 1.0, 0.0, // K
            0.5, 0.5, 0.5, 0.5, // V
        ];

        let output = simplified_attention(&config, &qkv, 1).unwrap();
        assert_eq!(output.len(), 4);
        // Single position: attention = softmax([score]) * V = 1.0 * V = V
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "output[{}] = {}, expected 0.5",
                i,
                v
            );
        }
    }

    #[test]
    fn test_simplified_attention_two_positions_causal() {
        let config = GpuModelConfig {
            hidden_dim: 2,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_dim: 4,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // Two positions: position 1 attends to both 0 and 1
        // Q: [[1, 0], [1, 0]]
        // K: [[1, 0], [0, 1]]
        // V: [[1, 0], [0, 1]]
        let qkv = vec![
            // Q (seq_len * hidden_dim)
            1.0, 0.0, 1.0, 0.0, // K (seq_len * hidden_dim)
            1.0, 0.0, 0.0, 1.0, // V (seq_len * hidden_dim)
            1.0, 0.0, 0.0, 1.0,
        ];

        let output = simplified_attention(&config, &qkv, 2).unwrap();
        assert_eq!(output.len(), 4);

        // Position 0: only attends to itself, output = V[0] = [1, 0]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!(output[1].abs() < 1e-5);
    }

    #[test]
    fn test_simplified_attention_multi_head() {
        let config = GpuModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 8,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // Single position with 2 heads (head_dim = 2)
        let qkv = vec![
            1.0, 1.0, 1.0, 1.0, // Q
            1.0, 1.0, 1.0, 1.0, // K
            0.5, 0.5, 0.5, 0.5, // V
        ];

        let output = simplified_attention(&config, &qkv, 1).unwrap();
        assert_eq!(output.len(), 4);
    }

    // === GpuModelConfig tests ===

    #[test]
    fn test_gpu_model_config_kv_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // kv_dim = head_dim * num_kv_heads = (4096/32) * 8 = 128 * 8 = 1024
        assert_eq!(config.kv_dim(), 1024);
    }

    #[test]
    fn test_gpu_model_config_qkv_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // qkv_dim = hidden_dim + 2 * kv_dim = 4096 + 2 * 1024 = 6144
        assert_eq!(config.qkv_dim(), 6144);
    }

    #[test]
    fn test_gpu_model_config_head_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // head_dim = hidden_dim / num_heads = 4096 / 32 = 128
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_gpu_model_config_gqa_ratio() {
        let config = GpuModelConfig {
            hidden_dim: 2048,
            num_heads: 32,
            num_kv_heads: 4,
            intermediate_dim: 5632,
            vocab_size: 32000,
            num_layers: 22,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // heads_per_kv = num_heads / num_kv_heads = 32 / 4 = 8
        let heads_per_kv = config.num_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 8);
    }

    #[test]
    fn test_gpu_model_config_mha_ratio() {
        // Standard MHA: num_kv_heads == num_heads
        let config = GpuModelConfig {
            hidden_dim: 768,
            num_heads: 12,
            num_kv_heads: 12,
            intermediate_dim: 3072,
            vocab_size: 30522,
            num_layers: 12,
            eps: 1e-12,
            rope_theta: 10000.0,
        };

        let heads_per_kv = config.num_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 1);
        assert_eq!(config.kv_dim(), config.hidden_dim);
    }
}
