//! GPU SIMD Operations Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - SIMD-accelerated compute primitives.
//!
//! ## Contents
//! - `scalar_softmax`, `simd_softmax` - Softmax implementations (IMP-038)
//! - `scalar_rope`, `simd_rope` - RoPE implementations (IMP-041)

// ============================================================================
// SIMD-accelerated operations (M18 - IMP-038)
// ============================================================================

/// Scalar softmax implementation (baseline for comparison)
///
/// Computes softmax using standard scalar operations.
#[must_use]
pub fn scalar_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// SIMD-accelerated softmax implementation (M18 - IMP-038)
///
/// Uses Trueno's SIMD operations for vectorized computation.
/// Falls back to scalar for unsupported sizes.
#[must_use]
pub fn simd_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max using SIMD via trueno
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) - exp is not SIMD accelerated
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();

    // Sum using trueno's SIMD sum
    let exp_vec = trueno::Vector::from_slice(&exp_vals);
    let sum = exp_vec.sum().unwrap_or_else(|_| exp_vals.iter().sum());

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

// ============================================================================
// Scalar and SIMD RoPE implementations (M19 - IMP-041)
// ============================================================================

/// Scalar RoPE (Rotary Position Embedding) implementation
///
/// Standard scalar implementation of rotary position embeddings.
/// Input shape: [seq_len * hidden_dim] flattened
#[must_use]
pub fn scalar_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let mut output = vec![0.0f32; input.len()];

    // Compute RoPE for each position
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // Apply rotary embedding to pairs of elements
            for i in 0..head_dim / 2 {
                let freq = 1.0 / theta.powf((2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx0 = head_start + i;
                let idx1 = head_start + i + head_dim / 2;

                if idx1 < input.len() {
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    output
}

/// SIMD-accelerated RoPE implementation (M19 - IMP-041)
///
/// Uses Trueno's SIMD operations for vectorized position encoding.
#[must_use]
pub fn simd_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let half_head = head_dim / 2;

    // Pre-compute frequency table (cache-friendly)
    let mut freqs: Vec<f32> = Vec::with_capacity(half_head);
    for i in 0..half_head {
        freqs.push(1.0 / theta.powf((2.0 * i as f32) / head_dim as f32));
    }

    let mut output = vec![0.0f32; input.len()];

    // Process each position using SIMD operations
    for pos in 0..seq_len {
        // Pre-compute angles for this position
        let angles: Vec<f32> = freqs.iter().map(|&f| pos as f32 * f).collect();
        let cos_vals: Vec<f32> = angles.iter().map(|&a| a.cos()).collect();
        let sin_vals: Vec<f32> = angles.iter().map(|&a| a.sin()).collect();

        // Use trueno vectors for batch operations
        let cos_vec = trueno::Vector::from_slice(&cos_vals);
        let sin_vec = trueno::Vector::from_slice(&sin_vals);

        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // Extract x0 and x1 halves
            let x0_slice = &input[head_start..head_start + half_head];
            let x1_slice = &input[head_start + half_head..head_start + head_dim];

            let x0_vec = trueno::Vector::from_slice(x0_slice);
            let x1_vec = trueno::Vector::from_slice(x1_slice);

            // Compute: out0 = x0 * cos - x1 * sin
            //          out1 = x0 * sin + x1 * cos
            let x0_cos = x0_vec.mul(&cos_vec).unwrap_or_else(|_| x0_vec.clone());
            let x1_sin = x1_vec.mul(&sin_vec).unwrap_or_else(|_| x1_vec.clone());
            let x0_sin = x0_vec.mul(&sin_vec).unwrap_or_else(|_| x0_vec.clone());
            let x1_cos = x1_vec.mul(&cos_vec).unwrap_or_else(|_| x1_vec.clone());

            let out0 = x0_cos
                .sub(&x1_sin)
                .unwrap_or_else(|_| trueno::Vector::from_slice(x0_slice));
            let out1 = x0_sin
                .add(&x1_cos)
                .unwrap_or_else(|_| trueno::Vector::from_slice(x1_slice));

            // Copy results to output
            output[head_start..head_start + half_head].copy_from_slice(out0.as_slice());
            output[head_start + half_head..head_start + head_dim].copy_from_slice(out1.as_slice());
        }
    }

    output
}
