//! ALB-095: APR Q4K GPU Adapter — quantized inference for Q4K APR models
//!
//! Uploads raw Q4K bytes from APR files directly to GPU via mmap.
//! 17 GB Q4K stays on GPU (fits 24 GB VRAM), dequantized per-kernel-call.
//!
//! # Architecture
//!
//! ```text
//! APR file (mmap) → get_tensor_bytes() → load_quantized_weights_with_type()
//!                                               ↓
//!                                        GPU Q4K GEMV kernels
//! ```
//!
//! # Five Whys
//!
//! 1. Why can't we run Qwen3-Coder-30B on GPU? — F32 weights = 120 GB, exceeds 24 GB VRAM
//! 2. Why F32? — AprF32ToGpuAdapter dequantizes Q4K→F32 on CPU before upload
//! 3. Why not keep Q4K on GPU? — No APR Q4K GPU adapter existed
//! 4. Why no adapter? — Q4K GPU path only existed for GGUF sources
//! 5. Why now? — ALB-093 created 17 GB Q4K APR; must enable GPU inference

#![allow(clippy::similar_names)]

#[cfg(feature = "cuda")]
use crate::apr::AprV2Model;
#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};

/// GGML Q4_K quantization type ID
#[cfg(feature = "cuda")]
const Q4K_TYPE: u32 = 12;
/// GGML Q6_K quantization type ID
#[cfg(feature = "cuda")]
const Q6K_TYPE: u32 = 14;
/// F32 type ID
#[cfg(feature = "cuda")]
const F32_TYPE: u32 = 0;

/// Model config parsed from APR metadata for Q4K inference.
#[derive(Debug, Clone)]
pub struct AprQ4KConfig {
    /// Hidden dimension (e.g., 3584 for Qwen3-Coder-30B).
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (GQA).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// FFN intermediate dimension (dense models).
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE theta for positional encoding.
    pub rope_theta: f64,
    /// Number of MoE experts (None for dense models).
    pub num_experts: Option<usize>,
    /// Top-k experts per token (None for dense models).
    pub num_experts_per_tok: Option<usize>,
    /// MoE expert intermediate dimension (None for dense models).
    pub moe_intermediate_size: Option<usize>,
}

/// Upload result with VRAM usage stats.
#[derive(Debug)]
pub struct UploadResult {
    /// Total bytes uploaded to GPU.
    pub total_bytes: usize,
    /// Total number of tensors uploaded.
    pub num_tensors: usize,
    /// Number of quantized (Q4K/Q6K) tensors.
    pub num_q4k_tensors: usize,
    /// Number of F32/F16 tensors (norms, embeddings).
    pub num_f32_tensors: usize,
}

/// Upload APR Q4K model weights to GPU.
///
/// Iterates the APR tensor index and uploads:
/// - Q4K/Q6K weights: raw bytes via `load_quantized_weights_with_type()`
/// - F32 weights (norms, embeddings): via `load_weights()` or `cache_rmsnorm_gamma()`
///
/// # Arguments
///
/// * `model` - APR model opened via mmap
/// * `executor` - CUDA executor with weight cache
///
/// # Returns
///
/// Upload statistics (total bytes, tensor counts)
/// Align `offset` up to the next multiple of `align` (must be power of 2).
#[cfg(feature = "cuda")]
const fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

/// ALB-098: Upload APR Q4K model weights to GPU using pool allocator.
///
/// Two-pass approach:
/// 1. Scan all tensors to compute total quantized bytes
/// 2. Allocate ONE large GPU buffer (pool)
/// 3. Copy each tensor into its 256-byte-aligned offset within the pool
///
/// This replaces 18,867 individual cuMemAlloc calls with 1, fixing
/// CUDA memory fragmentation OOM on MoE models.
///
/// F32/F16 tensors (norms, embeddings) still use individual allocations
/// since they're few (~100) and need separate type treatment.
#[cfg(feature = "cuda")]
pub fn upload_apr_q4k_weights(
    model: &AprV2Model,
    executor: &mut CudaExecutor,
) -> Result<UploadResult> {
    let tensor_names: Vec<String> = model.tensor_names().into_iter().map(String::from).collect();

    // Pass 1: Scan quantized tensors for total pool size
    let mut pool_size = 0usize;
    let mut quantized_entries: Vec<(String, u32, usize)> = Vec::new(); // (name, qtype, offset)

    for name in &tensor_names {
        let entry = model
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor disappeared: {name}"),
            })?;

        let bytes = model.get_tensor_bytes(name)?;
        let dtype = entry.dtype.as_str();

        let qtype = match dtype {
            "Q4_K" | "q4_k" => Some(Q4K_TYPE),
            "Q6_K" | "q6_k" => Some(Q6K_TYPE),
            other => crate::apr::dequant::dtype_to_ggml_qtype(other),
        };

        if let Some(qt) = qtype {
            let offset = pool_size;
            quantized_entries.push((name.clone(), qt, offset));
            pool_size = align_up(pool_size + bytes.len(), 256);
        }
    }

    let num_q4k = quantized_entries.len();
    println!(
        "  ALB-098: Pool allocator — {} quantized tensors, {:.1} GB in 1 cuMemAlloc",
        num_q4k,
        pool_size as f64 / 1e9
    );

    // Pass 2: Allocate pool and upload quantized weights
    let mut total_bytes = 0usize;

    if pool_size > 0 {
        executor
            .allocate_quantized_weight_pool(pool_size)
            .map_err(|e| RealizarError::GpuError {
                reason: format!(
                    "Failed to allocate {:.1} GB weight pool: {e}",
                    pool_size as f64 / 1e9
                ),
            })?;

        for (name, qtype, offset) in &quantized_entries {
            let bytes = model.get_tensor_bytes(name)?;
            let uploaded = executor
                .load_quantized_weights_pooled(name, bytes, *qtype, *offset)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {name} to pool: {e}"),
                })?;
            total_bytes += uploaded;
        }
    }

    // Pass 3: Upload non-quantized tensors (F32/F16 norms, embeddings) individually
    let mut num_f32 = 0usize;

    for name in &tensor_names {
        let entry = model
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor disappeared: {name}"),
            })?;

        let dtype = entry.dtype.as_str();

        // Skip quantized tensors (already in pool)
        match dtype {
            "Q4_K" | "q4_k" | "Q6_K" | "q6_k" => continue,
            other if crate::apr::dequant::dtype_to_ggml_qtype(other).is_some() => continue,
            _ => {},
        }

        let bytes = model.get_tensor_bytes(name)?;

        match dtype {
            "F32" | "f32" => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                if name.contains("norm") {
                    let uploaded = executor.cache_rmsnorm_gamma(name, &floats).map_err(|e| {
                        RealizarError::GpuError {
                            reason: format!("Failed to cache norm {name}: {e}"),
                        }
                    })?;
                    total_bytes += uploaded;
                } else {
                    let uploaded = executor.load_weights(name, &floats).map_err(|e| {
                        RealizarError::GpuError {
                            reason: format!("Failed to upload F32 {name}: {e}"),
                        }
                    })?;
                    total_bytes += uploaded;
                }
                num_f32 += 1;
            },
            "F16" | "f16" => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();

                if name.contains("norm") {
                    let uploaded = executor.cache_rmsnorm_gamma(name, &floats).map_err(|e| {
                        RealizarError::GpuError {
                            reason: format!("Failed to cache F16 norm {name}: {e}"),
                        }
                    })?;
                    total_bytes += uploaded;
                } else {
                    let uploaded = executor.load_weights(name, &floats).map_err(|e| {
                        RealizarError::GpuError {
                            reason: format!("Failed to upload F16 {name}: {e}"),
                        }
                    })?;
                    total_bytes += uploaded;
                }
                num_f32 += 1;
            },
            other => {
                eprintln!("[ALB-095] Skipping unsupported dtype {other} for {name}");
            },
        }
    }

    Ok(UploadResult {
        total_bytes,
        num_tensors: num_q4k + num_f32,
        num_q4k_tensors: num_q4k,
        num_f32_tensors: num_f32,
    })
}

/// Parse model config from APR metadata.
///
/// AprMetadata uses HuggingFace naming (`hidden_size`, `intermediate_size`).
/// MoE fields (`head_dim`, `num_experts`, etc.) come from the `extra` map
/// since they were added in ALB-094 and may not have typed struct fields yet.
#[cfg(feature = "cuda")]
pub fn parse_apr_q4k_config(model: &AprV2Model) -> Result<AprQ4KConfig> {
    let meta = model.metadata();

    let hidden_dim = meta.hidden_size.ok_or_else(|| RealizarError::FormatError {
        reason: "APR metadata missing hidden_size".to_string(),
    })?;
    let num_heads = meta.num_heads.ok_or_else(|| RealizarError::FormatError {
        reason: "APR metadata missing num_heads".to_string(),
    })?;
    let num_kv_heads = meta.num_kv_heads.unwrap_or(num_heads);
    let num_layers = meta.num_layers.ok_or_else(|| RealizarError::FormatError {
        reason: "APR metadata missing num_layers".to_string(),
    })?;
    let vocab_size = meta.vocab_size.ok_or_else(|| RealizarError::FormatError {
        reason: "APR metadata missing vocab_size".to_string(),
    })?;
    let intermediate_dim = meta.intermediate_size.unwrap_or(hidden_dim * 4);

    // head_dim: check extra map first, then infer from QKV weight size
    let head_dim = meta
        .extra
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            // ALB-095: Infer head_dim from layer 0 Q projection weight size
            // q_proj.weight shape: [num_heads * head_dim, hidden_dim] in Q4K
            // So: head_dim = q_out_dim / num_heads
            let q_tensor = model.get_tensor("model.layers.0.self_attn.q_proj.weight");
            if let Some(t) = q_tensor {
                let num_blocks = (t.size / 18) as usize;
                let total_elements = num_blocks * 32;
                let q_out_dim = total_elements / hidden_dim;
                let inferred = q_out_dim / num_heads;
                if inferred != hidden_dim / num_heads {
                    eprintln!(
                        "[ALB-095] Inferred head_dim={} from q_proj weight (vs default {})",
                        inferred,
                        hidden_dim / num_heads
                    );
                }
                inferred
            } else {
                hidden_dim / num_heads
            }
        });

    let eps = meta.rms_norm_eps.unwrap_or(1e-6);
    let rope_theta = meta.rope_theta.unwrap_or(10000.0);

    // MoE fields from metadata (ALB-094)
    // Try typed fields first (new APR files), then extra map (legacy), then infer from tensor names
    let mut num_experts = meta
        .extra
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let mut num_experts_per_tok = meta
        .extra
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let mut moe_intermediate_size = meta
        .extra
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    // ALB-095: Infer MoE config from tensor names when metadata is missing
    // (handles APR files created before MoE metadata was added to the converter)
    if num_experts.is_none() {
        let (inferred_experts, inferred_k, inferred_moe_inter) =
            infer_moe_config_from_tensors(model, hidden_dim);
        if let Some(n) = inferred_experts {
            num_experts = Some(n);
            num_experts_per_tok = num_experts_per_tok.or(inferred_k);
            moe_intermediate_size = moe_intermediate_size.or(inferred_moe_inter);
        }
    }

    Ok(AprQ4KConfig {
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        intermediate_dim,
        vocab_size,
        eps,
        rope_theta: rope_theta as f64,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
    })
}

/// ALB-095: Infer MoE config from tensor names when metadata is missing.
///
/// Scans layer 0's tensor names for expert patterns like
/// `model.layers.0.mlp.experts.{E}.gate_proj.weight` and infers:
/// - `num_experts`: max expert index + 1
/// - `num_experts_per_tok`: defaults to 8 (common for Qwen MoE)
/// - `moe_intermediate_size`: from expert weight shape (if available)
#[cfg(feature = "cuda")]
fn infer_moe_config_from_tensors(
    model: &AprV2Model,
    hidden_dim: usize,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    let names = model.tensor_names();

    // Count experts by scanning layer 0 tensor names
    let mut max_expert: Option<usize> = None;
    for name in &names {
        // Pattern: model.layers.0.mlp.experts.{E}.gate_proj.weight
        if let Some(rest) = name.strip_prefix("model.layers.0.mlp.experts.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(expert_id) = rest[..dot_pos].parse::<usize>() {
                    max_expert = Some(max_expert.map_or(expert_id, |m: usize| m.max(expert_id)));
                }
            }
        }
    }

    let num_experts = max_expert.map(|m| m + 1);

    if num_experts.is_none() {
        return (None, None, None);
    }

    // Default top-k to 8 (standard for Qwen MoE models)
    let num_experts_per_tok = Some(8);

    // Infer moe_intermediate_size from expert gate_proj weight shape
    // gate_proj.weight shape is [intermediate, hidden] in Q4K → byte count reveals intermediate_size
    let moe_intermediate: Option<usize> = model
        .get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        .map(|t| {
            // Q4K: each block is 32 elements packed into 18 bytes (2 bytes scale + 16 bytes data)
            // Total elements = (data_bytes / 18) * 32
            // intermediate = total_elements / hidden_dim
            let num_blocks = (t.size / 18) as usize;
            let total_elements = num_blocks * 32;
            total_elements / hidden_dim
        });

    let n = num_experts.unwrap();
    let k = num_experts_per_tok.unwrap();
    let inter = moe_intermediate.unwrap_or(0);
    eprintln!(
        "[ALB-095] Inferred MoE config from tensor names: {} experts, top-{}, intermediate={}",
        n, k, inter
    );

    (num_experts, num_experts_per_tok, moe_intermediate)
}

/// Q4K GEMV helper: run a cached Q4K weight GEMV via CudaExecutor.
///
/// # Arguments
/// * `executor` - CUDA executor
/// * `cache_key` - Name of the weight in the quantized cache
/// * `input` - Input vector [k]
/// * `n` - Output dimension
/// * `k` - Input dimension
///
/// # Returns
/// Output vector [n]
#[cfg(feature = "cuda")]
pub fn q4k_gemv(
    executor: &mut CudaExecutor,
    cache_key: &str,
    input: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; n];
    executor
        .q4k_gemv_cached(cache_key, input, &mut output, n as u32, k as u32)
        .map_err(|e| RealizarError::GpuError {
            reason: format!("Q4K GEMV failed for {cache_key}: {e}"),
        })?;
    Ok(output)
}

/// F32 GEMV helper for weights stored as F32 (embeddings, LM head).
///
/// Uses CPU matmul since these weights may not be in the quantized cache.
#[cfg(feature = "cuda")]
pub fn f32_matmul(weight: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    // weight layout: [out_dim, in_dim] row-major
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let offset = i * in_dim;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += weight[offset + j] * input[j];
        }
        output[i] = sum;
    }
    output
}

/// RMS norm on CPU.
#[cfg(feature = "cuda")]
fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut sum_sq = 0.0f32;
    for &v in input {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
    output
}

/// Apply RoPE (rotary position embedding) with NeoX interleaving.
#[cfg(feature = "cuda")]
fn apply_rope_neox(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    theta: f64,
    position: usize,
) {
    for h in 0..num_heads {
        let offset = h * head_dim;
        let half = head_dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = position as f64 * freq;
            let cos_val = angle.cos() as f32;
            let sin_val = angle.sin() as f32;

            let x0 = data[offset + i];
            let x1 = data[offset + half + i];
            data[offset + i] = x0 * cos_val - x1 * sin_val;
            data[offset + half + i] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// GQA attention for a single query token against cached K/V.
#[cfg(feature = "cuda")]
fn gqa_attention(
    q: &[f32],
    full_k: &[f32],
    full_v: &[f32],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let kv_h = h / q_per_kv;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores
        let mut scores = vec![0.0f32; kv_len];
        for pos in 0..kv_len {
            let k_offset = pos * kv_dim + kv_h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * full_k[k_offset + d];
            }
            scores[pos] = dot * scale;
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

        // Weighted sum of V
        let out_offset = h * head_dim;
        for pos in 0..kv_len {
            let v_offset = pos * kv_dim + kv_h * head_dim;
            let w = scores[pos];
            for d in 0..head_dim {
                output[out_offset + d] += w * full_v[v_offset + d];
            }
        }
    }

    output
}

/// SiLU activation (x * sigmoid(x))
#[cfg(feature = "cuda")]
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// MoE FFN forward pass for a single token using GPU Q4K GEMVs.
///
/// 1. Gate GEMV → softmax → top-k
/// 2. Per-expert: gate_proj/up_proj GEMV → SiLU × up → down_proj GEMV
/// 3. Weighted accumulation
#[cfg(feature = "cuda")]
fn moe_ffn_forward(
    executor: &mut CudaExecutor,
    hidden_state: &[f32],
    layer_idx: usize,
    hidden_dim: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate: usize,
) -> Result<Vec<f32>> {
    // Step 1: Gate GEMV (hidden_dim → num_experts)
    let gate_key = format!("model.layers.{layer_idx}.mlp.gate.weight");
    let logits = q4k_gemv(executor, &gate_key, hidden_state, num_experts, hidden_dim)?;

    // Softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; num_experts];
    let mut exp_sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        probs[i] = (l - max_logit).exp();
        exp_sum += probs[i];
    }
    for p in &mut probs {
        *p /= exp_sum;
    }

    // Top-k selection
    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k: Vec<(usize, f32)> = indexed.into_iter().take(num_experts_per_tok).collect();

    // Renormalize
    let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();
    let weights: Vec<f32> = if weight_sum > 0.0 {
        top_k.iter().map(|(_, w)| w / weight_sum).collect()
    } else {
        vec![1.0 / num_experts_per_tok as f32; num_experts_per_tok]
    };

    // Step 2: Per-expert SwiGLU
    let mut routed_out = vec![0.0f32; hidden_dim];

    for (idx, &(expert_id, _)) in top_k.iter().enumerate() {
        let gate_key = format!("model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight");
        let up_key = format!("model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight");
        let down_key = format!("model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight");

        // gate_proj: hidden_dim → moe_intermediate
        let gate_out = q4k_gemv(
            executor,
            &gate_key,
            hidden_state,
            moe_intermediate,
            hidden_dim,
        )?;
        // up_proj: hidden_dim → moe_intermediate
        let up_out = q4k_gemv(
            executor,
            &up_key,
            hidden_state,
            moe_intermediate,
            hidden_dim,
        )?;

        // SwiGLU: SiLU(gate) * up
        let mut act = vec![0.0f32; moe_intermediate];
        for i in 0..moe_intermediate {
            act[i] = silu(gate_out[i]) * up_out[i];
        }

        // down_proj: moe_intermediate → hidden_dim
        let down_out = q4k_gemv(executor, &down_key, &act, hidden_dim, moe_intermediate)?;

        // Weighted accumulation
        let w = weights[idx];
        for i in 0..hidden_dim {
            routed_out[i] += w * down_out[i];
        }
    }

    // Step 3: Shared expert (if present — Qwen3-Coder doesn't have shared experts)
    let shared_gate_key = format!("model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight");
    if executor.has_quantized_weights(&shared_gate_key) {
        let shared_up_key = format!("model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight");
        let shared_down_key =
            format!("model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight");

        let gate_out = q4k_gemv(
            executor,
            &shared_gate_key,
            hidden_state,
            moe_intermediate,
            hidden_dim,
        )?;
        let up_out = q4k_gemv(
            executor,
            &shared_up_key,
            hidden_state,
            moe_intermediate,
            hidden_dim,
        )?;

        let mut act = vec![0.0f32; moe_intermediate];
        for i in 0..moe_intermediate {
            act[i] = silu(gate_out[i]) * up_out[i];
        }

        let shared_out = q4k_gemv(
            executor,
            &shared_down_key,
            &act,
            hidden_dim,
            moe_intermediate,
        )?;

        // Check for shared expert gate
        let gate_weight_key = format!("model.layers.{layer_idx}.mlp.shared_expert_gate.weight");
        if executor.has_quantized_weights(&gate_weight_key) {
            let gate_logit = q4k_gemv(executor, &gate_weight_key, hidden_state, 1, hidden_dim)?;
            let gate_scale = 1.0 / (1.0 + (-gate_logit[0]).exp()); // sigmoid
            for i in 0..hidden_dim {
                routed_out[i] += gate_scale * shared_out[i];
            }
        } else {
            for i in 0..hidden_dim {
                routed_out[i] += shared_out[i];
            }
        }
    }

    Ok(routed_out)
}

/// Dense FFN forward pass using GPU Q4K GEMVs.
#[cfg(feature = "cuda")]
fn dense_ffn_forward(
    executor: &mut CudaExecutor,
    hidden_state: &[f32],
    layer_idx: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<Vec<f32>> {
    let gate_key = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
    let up_key = format!("model.layers.{layer_idx}.mlp.up_proj.weight");
    let down_key = format!("model.layers.{layer_idx}.mlp.down_proj.weight");

    let gate_out = q4k_gemv(
        executor,
        &gate_key,
        hidden_state,
        intermediate_dim,
        hidden_dim,
    )?;
    let up_out = q4k_gemv(
        executor,
        &up_key,
        hidden_state,
        intermediate_dim,
        hidden_dim,
    )?;

    let mut act = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        act[i] = silu(gate_out[i]) * up_out[i];
    }

    q4k_gemv(executor, &down_key, &act, hidden_dim, intermediate_dim)
}

/// Per-head RMS norm for QK-norm (Qwen3-style).
///
/// Applies RMSNorm independently to each head using a shared `[head_dim]` weight vector.
#[cfg(feature = "cuda")]
fn per_head_rms_norm(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    for h in 0..num_heads {
        let offset = h * head_dim;
        let head = &data[offset..offset + head_dim];
        let mut sum_sq = 0.0f32;
        for &v in head {
            sum_sq += v * v;
        }
        let rms = (sum_sq / head_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..head_dim {
            data[offset + i] *= inv_rms * weight[i];
        }
    }
}

/// Full forward pass for one token through the Q4K APR model.
///
/// Hybrid CPU/GPU: embedding, RMSNorm, RoPE, attention on CPU;
/// all projection GEMVs on GPU via Q4K kernels.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn forward_token_apr_q4k(
    executor: &mut CudaExecutor,
    config: &AprQ4KConfig,
    embedding_weight: &[f32],
    output_norm_weight: &[f32],
    layer_norm_weights: &[(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)],
    kv_cache_k: &mut Vec<Vec<f32>>, // [num_layers][kv_len * kv_dim]
    kv_cache_v: &mut Vec<Vec<f32>>,
    token_id: u32,
    position: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;

    // 1. Embedding lookup (CPU)
    let embed_offset = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = embedding_weight[embed_offset..embed_offset + hidden_dim].to_vec();

    // 2. Transformer layers
    for layer_idx in 0..config.num_layers {
        // 2a. Attention RMSNorm
        let normed = rms_norm(&hidden, &layer_norm_weights[layer_idx].0, config.eps);

        // 2b. Q/K/V projections (GPU Q4K GEMV)
        let q_key = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
        let k_key = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
        let v_key = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");

        let mut q = q4k_gemv(executor, &q_key, &normed, q_dim, hidden_dim)?;
        let mut k = q4k_gemv(executor, &k_key, &normed, kv_dim, hidden_dim)?;
        let v = q4k_gemv(executor, &v_key, &normed, kv_dim, hidden_dim)?;

        // 2b'. QK-norm (Qwen3-style): per-head RMSNorm on Q and K before RoPE
        if let Some(ref q_norm_w) = layer_norm_weights[layer_idx].2 {
            per_head_rms_norm(&mut q, num_heads, head_dim, q_norm_w, config.eps);
        }
        if let Some(ref k_norm_w) = layer_norm_weights[layer_idx].3 {
            per_head_rms_norm(&mut k, num_kv_heads, head_dim, k_norm_w, config.eps);
        }

        // 2c. RoPE
        apply_rope_neox(&mut q, num_heads, head_dim, config.rope_theta, position);
        apply_rope_neox(&mut k, num_kv_heads, head_dim, config.rope_theta, position);

        // 2d. Attention with KV cache
        let kv_len = kv_cache_k[layer_idx].len() / kv_dim + 1;

        // Build full K/V
        let mut full_k = kv_cache_k[layer_idx].clone();
        full_k.extend_from_slice(&k);
        let mut full_v = kv_cache_v[layer_idx].clone();
        full_v.extend_from_slice(&v);

        let attn_out = gqa_attention(
            &q,
            &full_k,
            &full_v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Update KV cache
        kv_cache_k[layer_idx].extend_from_slice(&k);
        kv_cache_v[layer_idx].extend_from_slice(&v);

        // 2e. Output projection (GPU Q4K GEMV)
        let o_key = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let attn_proj = q4k_gemv(executor, &o_key, &attn_out, hidden_dim, q_dim)?;

        // 2f. Residual
        for i in 0..hidden_dim {
            hidden[i] += attn_proj[i];
        }

        // 2g. FFN RMSNorm
        let ffn_normed = rms_norm(&hidden, &layer_norm_weights[layer_idx].1, config.eps);

        // 2h. FFN (MoE or dense)
        let ffn_out = if let (Some(num_experts), Some(k_experts), Some(moe_inter)) = (
            config.num_experts,
            config.num_experts_per_tok,
            config.moe_intermediate_size,
        ) {
            moe_ffn_forward(
                executor,
                &ffn_normed,
                layer_idx,
                hidden_dim,
                num_experts,
                k_experts,
                moe_inter,
            )?
        } else {
            dense_ffn_forward(
                executor,
                &ffn_normed,
                layer_idx,
                hidden_dim,
                config.intermediate_dim,
            )?
        };

        // 2i. Residual
        for i in 0..hidden_dim {
            hidden[i] += ffn_out[i];
        }
    }

    // 3. Final RMSNorm
    let final_normed = rms_norm(&hidden, output_norm_weight, config.eps);

    // 4. LM head (GPU Q4K GEMV)
    let lm_head_key = "model.embed_tokens.weight"; // tied embeddings
                                                   // Check if lm_head exists as a separate tensor
    let lm_head_output_key = "lm_head.weight";
    if executor.has_quantized_weights(lm_head_output_key) {
        q4k_gemv(
            executor,
            lm_head_output_key,
            &final_normed,
            config.vocab_size,
            hidden_dim,
        )
    } else if executor.has_quantized_weights(lm_head_key) {
        // Tied embeddings: use embed_tokens as LM head
        q4k_gemv(
            executor,
            lm_head_key,
            &final_normed,
            config.vocab_size,
            hidden_dim,
        )
    } else {
        // F32 LM head (embedding matmul on CPU)
        Ok(f32_matmul(
            embedding_weight,
            &final_normed,
            config.vocab_size,
            hidden_dim,
        ))
    }
}
