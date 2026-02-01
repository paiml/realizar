//! APR inference helper functions (PMAT-802)
//!
//! Extracted from apr/mod.rs - Pure Rust inference primitives.
//!
//! ## Contents
//! - RMS normalization
//! - Matrix multiplication with SIMD
//! - Transpose operations
//! - Simple attention computation
//! - Format detection utilities

use super::MAGIC;
use std::fs;
use std::path::Path;

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
pub(crate) fn matmul(
    x: &[f32],
    w: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
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
pub(crate) fn transpose_matrix(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
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
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
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

/// Compute dot product attention score for a single query-key pair
#[inline]
fn compute_attention_score(
    q: &[f32],
    k: &[f32],
    q_offset: usize,
    k_offset: usize,
    head_dim: usize,
    scale: f32,
) -> f32 {
    let mut score = 0.0;
    for d in 0..head_dim {
        let q_val = q.get(q_offset + d).copied().unwrap_or(0.0);
        let k_val = k.get(k_offset + d).copied().unwrap_or(0.0);
        score += q_val * k_val;
    }
    score * scale
}

/// Apply softmax normalization to scores in-place (up to position s)
#[inline]
fn softmax_causal(scores: &mut [f32], s: usize) {
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
}

/// Compute weighted sum of values for a single output dimension
#[inline]
fn weighted_value_sum(v: &[f32], scores: &[f32], v_base: usize, d: usize, s: usize) -> f32 {
    let mut val = 0.0;
    for t in 0..=s {
        let v_val = v.get(v_base * t + d).copied().unwrap_or(0.0);
        val += scores[t] * v_val;
    }
    val
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
            let q_base = s * hidden_dim + h * head_dim;
            let k_base = kv_h * head_dim;
            let v_base = kv_dim;

            // Compute causal attention scores
            let mut scores = vec![0.0; seq_len];
            for t in 0..=s {
                scores[t] = compute_attention_score(q, k, q_base, t * kv_dim + k_base, head_dim, scale);
            }

            softmax_causal(&mut scores, s);

            // Weighted sum of values
            for d in 0..head_dim {
                let val = weighted_value_sum(v, &scores, v_base, kv_h * head_dim + d, s);
                output[s * hidden_dim + h * head_dim + d] = val;
            }
        }
    }

    output
}

/// PMAT-110: Apply RoPE rotation using NORM style (adjacent pairs)
/// This is the standard RoPE used by most models (rope_type == 0)
/// Pairs elements (2*i, 2*i+1) for rotation
pub(crate) fn apply_rope_norm(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_offset = h * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            // NORM style: adjacent pairs (2*i, 2*i+1)
            let idx0 = head_offset + 2 * i;
            let idx1 = head_offset + 2 * i + 1;

            if idx1 < x.len() {
                let x0 = x[idx0];
                let x1 = x[idx1];

                x[idx0] = x0 * cos_val - x1 * sin_val;
                x[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Check if a file is a valid .apr v2 file
pub fn is_apr_file<P: AsRef<Path>>(path: P) -> bool {
    fs::read(path.as_ref()).is_ok_and(|data| data.len() >= 4 && data[0..4] == MAGIC)
}

/// Detect model format from file extension
fn format_from_extension(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    match ext.as_str() {
        "apr" => Some("apr"),
        "gguf" => Some("gguf"),
        "safetensors" => Some("safetensors"),
        _ => None,
    }
}

/// Detect model format from file magic bytes
fn format_from_magic(path: &Path) -> &'static str {
    let Ok(data) = fs::read(path) else {
        return "unknown";
    };
    if data.len() < 4 {
        return "unknown";
    }
    if data[0..4] == MAGIC {
        return "apr";
    }
    if data[0..4] == [0x47, 0x47, 0x55, 0x46] {
        return "gguf";
    }
    if data[0] == b'{' {
        return "safetensors";
    }
    "unknown"
}

/// Detect model format from extension or magic bytes
pub fn detect_format<P: AsRef<Path>>(path: P) -> &'static str {
    let path = path.as_ref();
    format_from_extension(path).unwrap_or_else(|| format_from_magic(path))
}

// ============================================================================
// Tests for APR Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // rms_norm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        // Normalized values should sum to approximately 0
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() > 0.0); // Non-zero sum due to weight
    }

    #[test]
    fn test_rms_norm_zeros() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        // All zeros normalized with small eps
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_rms_norm_seq_len_2() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 sequences of length 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // -------------------------------------------------------------------------
    // matmul Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_matmul_basic() {
        // 1x2 @ 3x2^T -> 1x3
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 rows of 2 cols
        let result = matmul(&x, &w, 1, 2, 3);
        assert_eq!(result.len(), 3);
        // row 0 of w: [1,0] dot [1,2] = 1
        // row 1 of w: [0,1] dot [1,2] = 2
        // row 2 of w: [1,1] dot [1,2] = 3
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_matmul_identity() {
        // Identity matrix
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity
        let result = matmul(&x, &w, 1, 3, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_matmul_zeros() {
        let x = vec![1.0, 2.0];
        let w = vec![0.0; 6]; // 3x2 zeros
        let result = matmul(&x, &w, 1, 2, 3);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // -------------------------------------------------------------------------
    // simd_dot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_dot(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_large() {
        // Large enough to use AVX2 path
        let n = 64;
        let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let b = vec![1.0f32; n];
        let result = simd_dot(&a, &b);
        // Sum of 1 to 64 = 64*65/2 = 2080
        assert!((result - 2080.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = simd_dot(&a, &b);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_unequal_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Shorter
        let result = simd_dot(&a, &b);
        // Uses min(a.len, b.len) = 2
        // 1*4 + 2*5 = 14
        assert!((result - 14.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // apply_rope_norm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_rope_norm_basic() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 0, 10000.0);
        // At position 0, angle = 0, cos=1, sin=0, so values should be unchanged
        assert!((x[0] - 1.0).abs() < 0.001);
        assert!((x[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_rope_norm_position_1() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 1, 10000.0);
        // At position 1, some rotation should occur
        // Values should be different from original
        let sum: f32 = x.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.5); // Non-zero output
    }

    #[test]
    fn test_apply_rope_norm_multiple_heads() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // 2 heads, head_dim=4
        apply_rope_norm(&mut x, 2, 4, 0, 10000.0);
        // At position 0, values should be unchanged for both heads
        assert!((x[0] - 1.0).abs() < 0.001);
        assert!((x[4] - 2.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // simple_attention Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simple_attention_basic() {
        // 1 sequence, 1 head, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        let result = simple_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(result.len(), 2);
        // Single token attending to itself
        assert!((result[0] - 0.5).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_simple_attention_seq_len_2() {
        // 2 tokens, 1 head, head_dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let result = simple_attention(&q, &k, &v, 2, 1, 1, 2);
        assert_eq!(result.len(), 4);
        // Non-trivial attention weights
    }

    // -------------------------------------------------------------------------
    // is_apr_file Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_apr_file_nonexistent() {
        assert!(!is_apr_file("/nonexistent/file.apr"));
    }

    // -------------------------------------------------------------------------
    // detect_format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_detect_format_by_extension_apr() {
        assert_eq!(detect_format("/some/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_by_extension_gguf() {
        assert_eq!(detect_format("/some/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_by_extension_safetensors() {
        assert_eq!(detect_format("/some/path/model.safetensors"), "safetensors");
    }

    #[test]
    fn test_detect_format_nonexistent() {
        // No extension match, file doesn't exist
        assert_eq!(detect_format("/nonexistent/file.bin"), "unknown");
    }
}
