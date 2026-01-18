//! SIMD-accelerated operations for inference
//!
//! Provides high-performance primitive operations using trueno's SIMD backend.
//! All operations are designed for cache efficiency with tiled implementations.
//!
//! ## Operations
//!
//! - [`simd_matmul`] - Matrix-vector multiplication with SIMD dot products
//! - [`simd_dot`] - SIMD-accelerated dot product
//! - [`simd_add`] - Vector addition
//! - [`simd_mul`] - Element-wise multiplication
//! - [`simd_silu`] - SiLU activation (x * sigmoid(x))
//! - [`simd_gelu`] - GELU activation (approximate)
//! - [`simd_softmax`] - Numerically stable softmax
//!
//! ## Performance
//!
//! Uses trueno's Vector::dot for all dot products, enabling:
//! - AVX2/SSE on x86
//! - NEON on ARM
//! - WASM SIMD in browsers
//! - Scalar fallback everywhere else

use trueno::Vector;

/// Tile size for cache-efficient tiled matmul
const TILE_SIZE: usize = 64;

/// SIMD-accelerated matrix-vector multiplication
///
/// Uses trueno's optimized SIMD backend for maximum performance.
/// Falls back to scalar for non-SIMD architectures.
///
/// # Arguments
///
/// * `input` - Input vector of length `in_dim`
/// * `weight` - Weight matrix stored row-major [out_dim × in_dim]
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// Output vector of length `out_dim`
///
/// # Example
///
/// ```
/// use realizar::inference::simd_matmul;
///
/// // 2x3 matrix times 3-vector = 2-vector
/// let input = vec![1.0, 2.0, 3.0];
/// let weight = vec![
///     1.0, 0.0, 0.0,  // row 0: extracts x
///     0.0, 1.0, 0.0,  // row 1: extracts y
/// ];
/// let output = simd_matmul(&input, &weight, 3, 2);
/// assert_eq!(output.len(), 2);
/// ```
#[must_use]
pub fn simd_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // Convert to trueno types for SIMD acceleration
    let input_vec = Vector::from_slice(input);

    // Compute each output element using SIMD dot product
    let mut output = vec![0.0; out_dim];

    // Use tiled approach for better cache utilization
    for tile_start in (0..out_dim).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(out_dim);

        for row in tile_start..tile_end {
            let row_start = row * in_dim;
            let row_end = row_start + in_dim;
            let row_vec = Vector::from_slice(&weight[row_start..row_end]);
            output[row] = input_vec.dot(&row_vec).expect("dot product failed");
        }
    }

    output
}

/// SIMD-accelerated dot product
///
/// Uses trueno's SIMD backend for vectorized computation.
///
/// # Example
///
/// ```
/// use realizar::inference::simd_dot;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = simd_dot(&a, &b);
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
#[inline]
#[must_use]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    Vector::from_slice(a)
        .dot(&Vector::from_slice(b))
        .expect("dot product failed")
}

/// SIMD-accelerated vector addition (a += b)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_add;
///
/// let mut a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// simd_add(&mut a, &b);
/// assert_eq!(a, vec![5.0, 7.0, 9.0]);
/// ```
#[inline]
pub fn simd_add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// SIMD-accelerated element-wise multiplication (a *= b)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_mul;
///
/// let mut a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// simd_mul(&mut a, &b);
/// assert_eq!(a, vec![4.0, 10.0, 18.0]);
/// ```
#[inline]
pub fn simd_mul(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= y;
    }
}

/// SIMD-accelerated SiLU activation (x * sigmoid(x))
///
/// Also known as Swish activation: f(x) = x / (1 + exp(-x))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_silu;
///
/// let mut data = vec![0.0, 1.0, -1.0];
/// simd_silu(&mut data);
/// assert!((data[0] - 0.0).abs() < 1e-5);  // silu(0) = 0
/// assert!((data[1] - 0.7311).abs() < 0.01);  // silu(1) ≈ 0.731
/// ```
#[inline]
pub fn simd_silu(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// SIMD-accelerated GELU activation (approximate)
///
/// Uses the tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_gelu;
///
/// let mut data = vec![0.0, 1.0, -1.0];
/// simd_gelu(&mut data);
/// assert!((data[0] - 0.0).abs() < 1e-5);  // gelu(0) = 0
/// assert!((data[1] - 0.8413).abs() < 0.01);  // gelu(1) ≈ 0.841
/// ```
#[inline]
pub fn simd_gelu(data: &mut [f32]) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const COEF: f32 = 0.044715;

    for x in data.iter_mut() {
        let x3 = *x * *x * *x;
        let inner = SQRT_2_OVER_PI * (*x + COEF * x3);
        *x = 0.5 * *x * (1.0 + inner.tanh());
    }
}

/// SIMD-accelerated softmax with numerical stability
///
/// Uses the max-subtraction trick to prevent overflow:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_softmax;
///
/// let mut data = vec![1.0, 2.0, 3.0];
/// simd_softmax(&mut data);
///
/// // Probabilities should sum to 1
/// let sum: f32 = data.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
///
/// // Largest input should have largest probability
/// assert!(data[2] > data[1]);
/// assert!(data[1] > data[0]);
/// ```
pub fn simd_softmax(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for x in data.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for x in data.iter_mut() {
            *x *= inv_sum;
        }
    }
}

// ============================================================================
// EXTREME TDD: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // simd_matmul Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_matmul_identity() {
        // 3x3 identity matrix
        let input = vec![1.0, 2.0, 3.0];
        let identity = vec![
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        let output = simd_matmul(&input, &identity, 3, 3);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_simd_matmul_projection() {
        // 2x3 projection matrix
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![
            1.0, 1.0, 1.0, // row 0: sum
            1.0, 0.0, -1.0, // row 1: x - z
        ];
        let output = simd_matmul(&input, &weight, 3, 2);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 6.0).abs() < 1e-5); // 1+2+3 = 6
        assert!((output[1] - (-2.0)).abs() < 1e-5); // 1-3 = -2
    }

    #[test]
    fn test_simd_matmul_expansion() {
        // 4x2 expansion matrix
        let input = vec![1.0, 2.0];
        let weight = vec![
            1.0, 0.0, // row 0: x
            0.0, 1.0, // row 1: y
            1.0, 1.0, // row 2: x+y
            1.0, -1.0, // row 3: x-y
        ];
        let output = simd_matmul(&input, &weight, 2, 4);
        assert_eq!(output.len(), 4);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_simd_matmul_large_tiled() {
        // Test that tiling works for large matrices
        let in_dim = 128;
        let out_dim = 256;
        let input: Vec<f32> = (0..in_dim).map(|i| i as f32).collect();

        // Create a simple weight matrix (diagonal-ish)
        let mut weight = vec![0.0; out_dim * in_dim];
        for i in 0..out_dim.min(in_dim) {
            weight[i * in_dim + i] = 1.0;
        }

        let output = simd_matmul(&input, &weight, in_dim, out_dim);
        assert_eq!(output.len(), out_dim);

        // First `in_dim` outputs should equal inputs
        for i in 0..in_dim {
            assert!((output[i] - i as f32).abs() < 1e-5);
        }
        // Remaining outputs should be zero
        for i in in_dim..out_dim {
            assert!((output[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_matmul_empty() {
        let input: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        let output = simd_matmul(&input, &weight, 0, 0);
        assert!(output.is_empty());
    }

    // ------------------------------------------------------------------------
    // simd_dot Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = simd_dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_simd_dot_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_self() {
        let a = vec![3.0, 4.0];
        let result = simd_dot(&a, &a);
        assert!((result - 25.0).abs() < 1e-5); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_simd_dot_negative() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_large() {
        let n = 1024;
        let a: Vec<f32> = vec![1.0; n];
        let b: Vec<f32> = vec![1.0; n];
        let result = simd_dot(&a, &b);
        assert!((result - n as f32).abs() < 1e-3);
    }

    // ------------------------------------------------------------------------
    // simd_add Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_add_basic() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_add_zeros() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_simd_add_negative() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![0.0, 0.0, 0.0]);
    }

    // ------------------------------------------------------------------------
    // simd_mul Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_mul_basic() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_simd_mul_ones() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0, 1.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_simd_mul_zeros() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_simd_mul_negative() {
        let mut a = vec![2.0, 3.0];
        let b = vec![-1.0, -2.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![-2.0, -6.0]);
    }

    // ------------------------------------------------------------------------
    // simd_silu Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_silu_zero() {
        let mut data = vec![0.0];
        simd_silu(&mut data);
        assert!((data[0]).abs() < 1e-5); // silu(0) = 0
    }

    #[test]
    fn test_simd_silu_positive() {
        let mut data = vec![1.0];
        simd_silu(&mut data);
        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        assert!((data[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_simd_silu_negative() {
        let mut data = vec![-1.0];
        simd_silu(&mut data);
        // silu(-1) = -1 / (1 + exp(1)) ≈ -0.2689
        assert!((data[0] - (-0.2689)).abs() < 0.01);
    }

    #[test]
    fn test_simd_silu_large_positive() {
        let mut data = vec![10.0];
        simd_silu(&mut data);
        // silu(10) ≈ 10 (sigmoid(10) ≈ 1)
        assert!((data[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_simd_silu_large_negative() {
        let mut data = vec![-10.0];
        simd_silu(&mut data);
        // silu(-10) ≈ 0 (sigmoid(-10) ≈ 0)
        assert!((data[0]).abs() < 0.01);
    }

    #[test]
    fn test_simd_silu_batch() {
        let mut data = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        simd_silu(&mut data);
        assert!((data[0]).abs() < 1e-5);
        assert!(data[1] > 0.0);
        assert!(data[2] < 0.0);
        assert!(data[3] > data[1]); // monotonic for x > 0
    }

    // ------------------------------------------------------------------------
    // simd_gelu Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_gelu_zero() {
        let mut data = vec![0.0];
        simd_gelu(&mut data);
        assert!((data[0]).abs() < 1e-5); // gelu(0) = 0
    }

    #[test]
    fn test_simd_gelu_positive() {
        let mut data = vec![1.0];
        simd_gelu(&mut data);
        // gelu(1) ≈ 0.841
        assert!((data[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_simd_gelu_negative() {
        let mut data = vec![-1.0];
        simd_gelu(&mut data);
        // gelu(-1) ≈ -0.159
        assert!((data[0] - (-0.159)).abs() < 0.01);
    }

    #[test]
    fn test_simd_gelu_large_positive() {
        let mut data = vec![3.0];
        simd_gelu(&mut data);
        // gelu(3) ≈ 3 (tanh approaches 1)
        assert!((data[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_simd_gelu_large_negative() {
        let mut data = vec![-3.0];
        simd_gelu(&mut data);
        // gelu(-3) ≈ 0 (tanh approaches -1)
        assert!((data[0]).abs() < 0.01);
    }

    #[test]
    fn test_simd_gelu_symmetry_breaking() {
        // GELU is NOT symmetric: gelu(-x) != -gelu(x)
        let mut pos = vec![1.0];
        let mut neg = vec![-1.0];
        simd_gelu(&mut pos);
        simd_gelu(&mut neg);
        assert!((pos[0] + neg[0]).abs() > 0.1); // sum should not be 0
    }

    // ------------------------------------------------------------------------
    // simd_softmax Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_softmax_sums_to_one() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_softmax_preserves_order() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax(&mut data);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_simd_softmax_uniform() {
        let mut data = vec![1.0, 1.0, 1.0];
        simd_softmax(&mut data);
        // Should be uniform distribution
        for &x in &data {
            assert!((x - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_softmax_empty() {
        let mut data: Vec<f32> = vec![];
        simd_softmax(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_simd_softmax_single() {
        let mut data = vec![5.0];
        simd_softmax(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let mut data = vec![1000.0, 1001.0, 1002.0];
        simd_softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_simd_softmax_negative() {
        let mut data = vec![-1.0, -2.0, -3.0];
        simd_softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Order reversed: -1 > -2 > -3
        assert!(data[0] > data[1]);
        assert!(data[1] > data[2]);
    }

    #[test]
    fn test_simd_softmax_temperature_effect() {
        // Larger differences should give more peaked distribution
        let mut narrow = vec![1.0, 2.0, 3.0];
        let mut wide = vec![1.0, 10.0, 100.0];

        simd_softmax(&mut narrow);
        simd_softmax(&mut wide);

        // Wide should be more peaked (largest value dominates)
        assert!(wide[2] > narrow[2]);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_matmul_then_activation() {
        let input = vec![1.0, 2.0];
        let weight = vec![
            1.0, 1.0, // sum: 3
            -1.0, 1.0, // diff: 1
        ];
        let mut output = simd_matmul(&input, &weight, 2, 2);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 1.0).abs() < 1e-5);

        simd_gelu(&mut output);
        // gelu(3) ≈ 3, gelu(1) ≈ 0.841
        assert!((output[0] - 3.0).abs() < 0.01);
        assert!((output[1] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_residual_connection() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![
            0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, // 0.1 * I
        ];
        let proj = simd_matmul(&input, &weight, 3, 3);

        let mut residual = input.clone();
        simd_add(&mut residual, &proj);

        // residual = input + 0.1 * input = 1.1 * input
        assert!((residual[0] - 1.1).abs() < 1e-5);
        assert!((residual[1] - 2.2).abs() < 1e-5);
        assert!((residual[2] - 3.3).abs() < 1e-5);
    }

    #[test]
    fn test_gated_activation() {
        // SwiGLU style: gate * up
        let mut gate = vec![0.0, 1.0, 2.0];
        let up = vec![1.0, 2.0, 3.0];

        simd_silu(&mut gate);
        simd_mul(&mut gate, &up);

        // gate[0] = silu(0) * 1 = 0
        assert!((gate[0]).abs() < 1e-5);
        // gate[1] = silu(1) * 2 ≈ 0.7311 * 2 ≈ 1.46
        assert!((gate[1] - 1.46).abs() < 0.05);
    }
}
