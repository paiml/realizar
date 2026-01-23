//! BPE Tokenizer for APR models (PMAT-802)
//!
//! Byte Pair Encoding tokenizer supporting APR v2 format models.

use std::collections::HashMap;
use crate::error::{RealizarError, Result};
use super::AprV2Model;

/// BPE Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string to ID mapping
    pub token_to_id: HashMap<String, u32>,
    /// ID to token string mapping (index = ID)
    pub id_to_token: Vec<String>,
    /// BPE merge rules (first, second) pairs
    pub merge_rules: Vec<(String, String)>,
    /// Beginning-of-sequence token ID
    pub bos_id: Option<u32>,
    /// End-of-sequence token ID
    pub eos_id: Option<u32>,
}

impl BpeTokenizer {
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        bpe_encode(text, &self.token_to_id, &self.merge_rules)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }
}

/// Byte-level BPE encoding
pub(crate) fn bpe_encode(text: &str, vocab: &HashMap<String, u32>, merges: &[(String, String)]) -> Vec<u32> {
    // Convert text to byte-level tokens (GPT-2/Qwen style)
    // Each byte maps to a special unicode char in range U+0100-U+01FF or similar
    let mut tokens: Vec<String> = text
        .chars()
        .map(|c| {
            // Convert character to byte-level BPE token
            // Space becomes Ġ (U+0120 = 288), newline becomes Ċ, etc.
            if c == ' ' {
                "Ġ".to_string()
            } else if c == '\n' {
                "Ċ".to_string()
            } else if c == '\t' {
                "ĉ".to_string()
            } else if c.is_ascii() {
                c.to_string()
            } else {
                // For non-ASCII, encode as bytes
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.chars()
                    .map(|byte_char| byte_to_bpe_char(byte_char as u8))
                    .collect()
            }
        })
        .collect();

    // Apply BPE merges iteratively
    for (first, second) in merges {
        let merged = format!("{}{}", first, second);
        loop {
            let mut found = false;
            let mut i = 0;
            while i + 1 < tokens.len() {
                if &tokens[i] == first && &tokens[i + 1] == second {
                    tokens[i].clone_from(&merged);
                    tokens.remove(i + 1);
                    found = true;
                }
                i += 1;
            }
            if !found {
                break;
            }
        }
    }

    // Convert tokens to IDs
    tokens
        .iter()
        .filter_map(|t| vocab.get(t).copied())
        .collect()
}

/// Convert byte to BPE character representation
pub fn byte_to_bpe_char(b: u8) -> String {
    // GPT-2/Qwen byte-level BPE uses specific unicode mappings
    // This is a simplified version - real tokenizers use a full byte-to-unicode table
    match b {
        b' ' => "Ġ".to_string(),
        b'\n' => "Ċ".to_string(),
        b'\t' => "ĉ".to_string(),
        _ if b.is_ascii_graphic() || b.is_ascii_alphanumeric() => (b as char).to_string(),
        _ => format!("<0x{:02X}>", b),
    }
}

/// RMS normalization
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = x.len() / hidden_dim;
    let mut output = Vec::with_capacity(x.len());

    for s in 0..seq_len {
        let start = s * hidden_dim;
        let slice = &x[start..start + hidden_dim];

        // Compute RMS
        let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        // Normalize and scale
        for (i, &v) in slice.iter().enumerate() {
            output.push((v / rms) * weight.get(i).copied().unwrap_or(1.0));
        }
    }
    output
}

/// Matrix multiplication with SIMD dot products
/// [seq, in_dim] @ [out_dim, in_dim]^T -> [seq, out_dim]
pub(crate) fn matmul(x: &[f32], w: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; seq_len * out_dim];

    for s in 0..seq_len {
        let x_start = s * in_dim;
        let x_end = x_start + in_dim;
        if x_end > x.len() {
            continue; // Skip if out of bounds
        }
        let x_row = &x[x_start..x_end];

        for o in 0..out_dim {
            let w_start = o * in_dim;
            let w_end = w_start + in_dim;
            if w_end > w.len() {
                continue; // Skip if out of bounds
            }
            let w_row = &w[w_start..w_end];
            // SIMD dot product
            output[s * out_dim + o] = simd_dot(x_row, w_row);
        }
    }
    output
}

/// Transpose a matrix from [rows, cols] to [cols, rows] for GEMM compatibility.
/// Weight matrices are stored as [out_dim, in_dim] but GEMM needs [in_dim, out_dim].
#[cfg(feature = "cuda")]
fn transpose_matrix(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            // m[r, c] -> transposed[c, r]
            let src_idx = r * cols + c;
            let dst_idx = c * rows + r;
            if src_idx < m.len() && dst_idx < transposed.len() {
                transposed[dst_idx] = m[src_idx];
            }
        }
    }
    transposed
}

/// SIMD-accelerated dot product
#[inline]
pub(crate) fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature is runtime-checked above, simd_dot_avx2 requires AVX2
            return unsafe { simd_dot_avx2(a, b) };
        }
    }
    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    let chunks = n / 8;

    // SAFETY: This entire fn is unsafe with target_feature(avx2, fma)
    // All intrinsics are safe to call given the target_feature guarantee
    // The unsafe block is required for Rust 2024 edition compliance
    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(av, bv, sum);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder (scalar)
        for i in (chunks * 8)..n {
            result += a.get(i).copied().unwrap_or(0.0) * b.get(i).copied().unwrap_or(0.0);
        }

        result
    }
}

/// Simplified multi-head attention (no RoPE, causal mask)
pub(crate) fn simple_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; seq_len * hidden_dim];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            // Compute attention scores for this head
            let mut scores = vec![0.0; seq_len];
            for t in 0..=s {
                // Causal: only attend to past
                let mut score = 0.0;
                for d in 0..head_dim {
                    let q_val = q
                        .get(s * hidden_dim + h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
                    let k_val = k
                        .get(t * kv_dim + kv_h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
                    score += q_val * k_val;
                }
                scores[t] = score * scale;
            }

            // Softmax
            let max_score = scores[..=s]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for score in &mut scores[..=s] {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores[..=s] {
                *score /= sum;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut val = 0.0;
                for t in 0..=s {
                    let v_val = v
                        .get(t * kv_dim + kv_h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
                    val += scores[t] * v_val;
                }
                output[s * hidden_dim + h * head_dim + d] = val;
            }
        }
    }

    output
}

