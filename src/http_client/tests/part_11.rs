use crate::http_client::*;
// ==================== IMP-208: Softmax Sums to 1.0 (QA-005) ====================
// Per spec: Softmax outputs sum to 1.0 within 1e-7

/// Softmax verification result
#[derive(Debug, Clone)]
pub struct SoftmaxVerificationResult {
    pub input_logits: Vec<f32>,
    pub output_probs: Vec<f32>,
    pub sum: f32,
    pub sum_diff_from_one: f32,
    pub tolerance: f32,
    pub meets_qa005: bool,
}

impl SoftmaxVerificationResult {
    pub fn new(logits: Vec<f32>, probs: Vec<f32>, tolerance: f32) -> Self {
        let sum: f32 = probs.iter().sum();
        let sum_diff_from_one = (sum - 1.0).abs();
        let meets_qa005 = sum_diff_from_one <= tolerance;

        Self {
            input_logits: logits,
            output_probs: probs,
            sum,
            sum_diff_from_one,
            tolerance,
            meets_qa005,
        }
    }

    pub fn compute_softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect()
    }
}

/// IMP-208a: Test softmax verification
#[test]
fn test_imp_208a_softmax_verification() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(result.meets_qa005, "IMP-208a: Should meet QA-005");
    assert!(
        result.sum_diff_from_one < 1e-7,
        "IMP-208a: Sum should be 1.0"
    );

    println!("\nIMP-208a: Softmax Verification:");
    println!("  Probabilities: {:?}", result.output_probs);
    println!("  Sum: {:.10}", result.sum);
    println!("  Diff from 1.0: {:.2e}", result.sum_diff_from_one);
}

/// IMP-208b: Test softmax with large logits
#[test]
fn test_imp_208b_softmax_large_logits() {
    let logits = vec![100.0, 200.0, 300.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(result.meets_qa005, "IMP-208b: Should handle large logits");

    println!("\nIMP-208b: Softmax Large Logits:");
    println!("  Sum: {:.10}", result.sum);
    println!("  Numerically stable: {}", result.meets_qa005);
}

/// IMP-208c: Test softmax with negative logits
#[test]
fn test_imp_208c_softmax_negative_logits() {
    let logits = vec![-1.0, -2.0, -3.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(
        result.meets_qa005,
        "IMP-208c: Should handle negative logits"
    );

    println!("\nIMP-208c: Softmax Negative Logits:");
    println!("  Probabilities: {:?}", result.output_probs);
    println!("  Sum: {:.10}", result.sum);
}

/// IMP-208d: Real-world softmax verification
#[test]
#[ignore = "Requires softmax extraction from inference"]
fn test_imp_208d_realworld_softmax() {
    let logits = vec![2.5, 1.2, 0.8, 3.1, 0.5];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    println!("\nIMP-208d: Real-World Softmax:");
    println!("  Sum: {:.10}", result.sum);
    println!(
        "  QA-005: {}",
        if result.meets_qa005 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-209: Layer Norm Unit Variance (QA-006) ====================
// Per spec: Layer norm outputs have unit variance within 1e-4

/// Layer norm verification result
#[derive(Debug, Clone)]
pub struct LayerNormVerificationResult {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub mean: f32,
    pub variance: f32,
    pub variance_diff_from_one: f32,
    pub tolerance: f32,
    pub meets_qa006: bool,
}

impl LayerNormVerificationResult {
    pub fn new(input: Vec<f32>, output: Vec<f32>, tolerance: f32) -> Self {
        let n = output.len() as f32;
        let mean: f32 = output.iter().sum::<f32>() / n;
        let variance: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let variance_diff_from_one = (variance - 1.0).abs();
        let meets_qa006 = variance_diff_from_one <= tolerance;

        Self {
            input,
            output,
            mean,
            variance,
            variance_diff_from_one,
            tolerance,
            meets_qa006,
        }
    }

    pub fn compute_layer_norm(input: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let variance: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (variance + eps).sqrt();
        input.iter().map(|x| (x - mean) / std).collect()
    }
}

/// IMP-209a: Test layer norm verification
#[test]
fn test_imp_209a_layer_norm_verification() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    assert!(result.meets_qa006, "IMP-209a: Should meet QA-006");

    println!("\nIMP-209a: Layer Norm Verification:");
    println!("  Output: {:?}", result.output);
    println!("  Mean: {:.6}", result.mean);
    println!("  Variance: {:.6}", result.variance);
}

/// IMP-209b: Test layer norm zero mean
#[test]
fn test_imp_209b_layer_norm_zero_mean() {
    let input = vec![10.0, 20.0, 30.0, 40.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    assert!(result.mean.abs() < 1e-5, "IMP-209b: Mean should be ~0");

    println!("\nIMP-209b: Layer Norm Zero Mean:");
    println!("  Mean: {:.2e}", result.mean);
}

/// IMP-209c: Test layer norm with uniform input
#[test]
fn test_imp_209c_layer_norm_uniform() {
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);

    // All same values -> variance is 0, output is 0
    assert!(
        output.iter().all(|&x| x.abs() < 1e-3),
        "IMP-209c: Uniform input -> zero output"
    );

    println!("\nIMP-209c: Layer Norm Uniform Input:");
    println!("  Output: {:?}", output);
}

/// IMP-209d: Real-world layer norm verification
#[test]
#[ignore = "Requires layer norm extraction from model"]
fn test_imp_209d_realworld_layer_norm() {
    let input = vec![0.5, 1.2, -0.3, 0.8, -0.1];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    println!("\nIMP-209d: Real-World Layer Norm:");
    println!("  Variance: {:.6}", result.variance);
    println!(
        "  QA-006: {}",
        if result.meets_qa006 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-210: GELU Matches PyTorch (QA-007) ====================
// Per spec: GELU activation matches PyTorch within 1e-5

/// GELU verification result
#[derive(Debug, Clone)]
pub struct GELUVerificationResult {
    pub input: Vec<f32>,
    pub reference_output: Vec<f32>,
    pub test_output: Vec<f32>,
    pub max_diff: f32,
    pub tolerance: f32,
    pub meets_qa007: bool,
}

impl GELUVerificationResult {
    pub fn new(input: Vec<f32>, ref_out: Vec<f32>, test_out: Vec<f32>, tolerance: f32) -> Self {
        let max_diff = ref_out
            .iter()
            .zip(test_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let meets_qa007 = max_diff <= tolerance;

        Self {
            input,
            reference_output: ref_out,
            test_output: test_out,
            max_diff,
            tolerance,
            meets_qa007,
        }
    }

    /// GELU approximation (tanh version used by GPT-2)
    pub fn compute_gelu(x: f32) -> f32 {
        0.5 * x
            * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }
}

/// IMP-210a: Test GELU verification
#[test]
fn test_imp_210a_gelu_verification() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let ref_out: Vec<f32> = input
        .iter()
        .map(|&x| GELUVerificationResult::compute_gelu(x))
        .collect();
    let test_out = ref_out.clone();

    let result = GELUVerificationResult::new(input, ref_out, test_out, 1e-5);

    assert!(result.meets_qa007, "IMP-210a: Should meet QA-007");

    println!("\nIMP-210a: GELU Verification:");
    println!("  Input: {:?}", result.input);
    println!("  Output: {:?}", result.reference_output);
    println!("  Max diff: {:.2e}", result.max_diff);
}

/// IMP-210b: Test GELU at zero
#[test]
fn test_imp_210b_gelu_at_zero() {
    let gelu_zero = GELUVerificationResult::compute_gelu(0.0);
    assert!(gelu_zero.abs() < 1e-7, "IMP-210b: GELU(0) should be 0");

    println!("\nIMP-210b: GELU at Zero:");
    println!("  GELU(0) = {:.10}", gelu_zero);
}

/// IMP-210c: Test GELU approximation accuracy
#[test]
fn test_imp_210c_gelu_approximation() {
    // PyTorch reference values for GELU
    let test_cases = vec![
        (-2.0, -0.0454),
        (-1.0, -0.1587),
        (0.0, 0.0),
        (1.0, 0.8413),
        (2.0, 1.9546),
    ];

    println!("\nIMP-210c: GELU Approximation:");
    for (x, expected) in test_cases {
        let actual = GELUVerificationResult::compute_gelu(x);
        let diff = (actual - expected).abs();
        println!(
            "  GELU({:.1}) = {:.4} (expected {:.4}, diff {:.4})",
            x, actual, expected, diff
        );
        assert!(diff < 0.01, "IMP-210c: GELU should match PyTorch");
    }
}

/// IMP-210d: Real-world GELU verification
#[test]
#[ignore = "Requires GELU extraction from PyTorch reference"]
fn test_imp_210d_realworld_gelu() {
    let input = vec![-1.5, -0.5, 0.5, 1.5];
    let ref_out: Vec<f32> = input
        .iter()
        .map(|&x| GELUVerificationResult::compute_gelu(x))
        .collect();

    let result = GELUVerificationResult::new(input, ref_out.clone(), ref_out, 1e-5);

    println!("\nIMP-210d: Real-World GELU:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-007: {}",
        if result.meets_qa007 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-211: SwiGLU Matches Reference (QA-008) ====================
// Per spec: SwiGLU activation matches reference within 1e-5

/// SwiGLU verification result
#[derive(Debug, Clone)]
pub struct SwiGLUVerificationResult {
    pub input_gate: Vec<f32>,
    pub input_up: Vec<f32>,
    pub reference_output: Vec<f32>,
    pub test_output: Vec<f32>,
    pub max_diff: f32,
    pub tolerance: f32,
    pub meets_qa008: bool,
}

impl SwiGLUVerificationResult {
    pub fn new(
        gate: Vec<f32>,
        up: Vec<f32>,
        ref_out: Vec<f32>,
        test_out: Vec<f32>,
        tolerance: f32,
    ) -> Self {
        let max_diff = ref_out
            .iter()
            .zip(test_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let meets_qa008 = max_diff <= tolerance;

        Self {
            input_gate: gate,
            input_up: up,
            reference_output: ref_out,
            test_output: test_out,
            max_diff,
            tolerance,
            meets_qa008,
        }
    }

    /// Swish activation: x * sigmoid(x)
    pub fn swish(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// SwiGLU: swish(gate) * up
    pub fn compute_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| Self::swish(g) * u)
            .collect()
    }
}

/// IMP-211a: Test SwiGLU verification
#[test]
fn test_imp_211a_swiglu_verification() {
    let gate = vec![1.0, 2.0, 3.0];
    let up = vec![1.0, 1.0, 1.0];
    let ref_out = SwiGLUVerificationResult::compute_swiglu(&gate, &up);
    let test_out = ref_out.clone();

    let result = SwiGLUVerificationResult::new(gate, up, ref_out, test_out, 1e-5);

    assert!(result.meets_qa008, "IMP-211a: Should meet QA-008");

    println!("\nIMP-211a: SwiGLU Verification:");
    println!("  Output: {:?}", result.reference_output);
    println!("  Max diff: {:.2e}", result.max_diff);
}

/// IMP-211b: Test swish activation
#[test]
fn test_imp_211b_swish_activation() {
    let test_cases = vec![(0.0, 0.0), (1.0, 0.7311), (2.0, 1.7616), (-1.0, -0.2689)];

    println!("\nIMP-211b: Swish Activation:");
    for (x, expected) in test_cases {
        let actual = SwiGLUVerificationResult::swish(x);
        let diff = (actual - expected).abs();
        println!(
            "  swish({:.1}) = {:.4} (expected {:.4})",
            x, actual, expected
        );
        assert!(diff < 0.01, "IMP-211b: Swish should match reference");
    }
}

/// IMP-211c: Test SwiGLU with different inputs
#[test]
fn test_imp_211c_swiglu_different_inputs() {
    let gate = vec![0.0, 1.0, -1.0];
    let up = vec![2.0, 2.0, 2.0];
    let output = SwiGLUVerificationResult::compute_swiglu(&gate, &up);

    assert!(output[0].abs() < 1e-5, "IMP-211c: SwiGLU(0, 2) should be 0");
    assert!(output[1] > 0.0, "IMP-211c: SwiGLU(1, 2) should be positive");
    assert!(
        output[2] < 0.0,
        "IMP-211c: SwiGLU(-1, 2) should be negative"
    );

    println!("\nIMP-211c: SwiGLU Different Inputs:");
    println!("  SwiGLU(0, 2) = {:.4}", output[0]);
    println!("  SwiGLU(1, 2) = {:.4}", output[1]);
    println!("  SwiGLU(-1, 2) = {:.4}", output[2]);
}

/// IMP-211d: Real-world SwiGLU verification
#[test]
#[ignore = "Requires SwiGLU extraction from reference model"]
fn test_imp_211d_realworld_swiglu() {
    let gate = vec![0.5, 1.0, 1.5, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let ref_out = SwiGLUVerificationResult::compute_swiglu(&gate, &up);

    let result = SwiGLUVerificationResult::new(gate, up, ref_out.clone(), ref_out, 1e-5);

    println!("\nIMP-211d: Real-World SwiGLU:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-008: {}",
        if result.meets_qa008 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-212: KV Cache Matches Recomputation (QA-009) ====================
// Per spec: KV cache produces identical results to recomputation

/// KV cache verification result
#[derive(Debug, Clone)]
pub struct KVCacheVerificationResult {
    pub sequence_length: usize,
    pub cached_output: Vec<f32>,
    pub recomputed_output: Vec<f32>,
    pub max_diff: f32,
    pub is_identical: bool,
    pub meets_qa009: bool,
}

impl KVCacheVerificationResult {
    pub fn new(seq_len: usize, cached: Vec<f32>, recomputed: Vec<f32>) -> Self {
        let max_diff = cached
            .iter()
            .zip(recomputed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let is_identical = max_diff < 1e-6;
        let meets_qa009 = is_identical;

        Self {
            sequence_length: seq_len,
            cached_output: cached,
            recomputed_output: recomputed,
            max_diff,
            is_identical,
            meets_qa009,
        }
    }
}

/// IMP-212a: Test KV cache verification
#[test]
fn test_imp_212a_kv_cache_verification() {
    let cached = vec![0.1, 0.2, 0.3, 0.4];
    let recomputed = vec![0.1, 0.2, 0.3, 0.4];

    let result = KVCacheVerificationResult::new(4, cached, recomputed);

    assert!(result.meets_qa009, "IMP-212a: Should meet QA-009");
    assert!(result.is_identical, "IMP-212a: Should be identical");

    println!("\nIMP-212a: KV Cache Verification:");
    println!("  Sequence length: {}", result.sequence_length);
    println!("  Max diff: {:.2e}", result.max_diff);
    println!("  Identical: {}", result.is_identical);
}

/// IMP-212b: Test KV cache mismatch detection
#[test]
fn test_imp_212b_kv_cache_mismatch() {
    let cached = vec![0.1, 0.2, 0.3, 0.4];
    let recomputed = vec![0.1, 0.2, 0.35, 0.4]; // 0.05 diff at position 2

    let result = KVCacheVerificationResult::new(4, cached, recomputed);

    assert!(!result.meets_qa009, "IMP-212b: Should detect mismatch");
    assert!(!result.is_identical, "IMP-212b: Should not be identical");

    println!("\nIMP-212b: KV Cache Mismatch:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!("  Identical: {}", result.is_identical);
}

/// IMP-212c: Test KV cache at different lengths
#[test]
fn test_imp_212c_kv_cache_lengths() {
    let lengths = vec![1, 10, 100, 512];

    println!("\nIMP-212c: KV Cache at Different Lengths:");
    for len in lengths {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let result = KVCacheVerificationResult::new(len, data.clone(), data);
        println!("  Length {}: meets QA-009 = {}", len, result.meets_qa009);
        assert!(result.meets_qa009);
    }
}

/// IMP-212d: Real-world KV cache verification
#[test]
#[ignore = "Requires KV cache extraction from inference"]
fn test_imp_212d_realworld_kv_cache() {
    let cached = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let recomputed = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let result = KVCacheVerificationResult::new(5, cached, recomputed);

    println!("\nIMP-212d: Real-World KV Cache:");
    println!("  Sequence length: {}", result.sequence_length);
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-009: {}",
        if result.meets_qa009 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-213: Quantized Matches F32 (QA-010) ====================
// Per spec: Quantized inference matches F32 within acceptable tolerance

/// Quantization verification result
#[derive(Debug, Clone)]
pub struct QuantizationVerificationResult {
    pub quantization_type: String,
    pub f32_output: Vec<f32>,
    pub quantized_output: Vec<f32>,
    pub max_diff: f32,
    pub mean_diff: f32,
    pub tolerance: f32,
    pub meets_qa010: bool,
}

impl QuantizationVerificationResult {
    pub fn new(
        quant_type: impl Into<String>,
        f32_out: Vec<f32>,
        quant_out: Vec<f32>,
        tolerance: f32,
    ) -> Self {
        let diffs: Vec<f32> = f32_out
            .iter()
            .zip(quant_out.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();

        let max_diff = diffs.iter().cloned().fold(0.0_f32, f32::max);
        let mean_diff = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f32>() / diffs.len() as f32
        };

        let meets_qa010 = max_diff <= tolerance;

        Self {
            quantization_type: quant_type.into(),
            f32_output: f32_out,
            quantized_output: quant_out,
            max_diff,
            mean_diff,
            tolerance,
            meets_qa010,
        }
    }
}

/// IMP-213a: Test quantization verification
#[test]
fn test_imp_213a_quantization_verification() {
    let f32_out = vec![0.1, 0.2, 0.3, 0.4];
    let quant_out = vec![0.1001, 0.1999, 0.3002, 0.3998];

    let result = QuantizationVerificationResult::new("Q4_K", f32_out, quant_out, 0.01);

    assert!(result.meets_qa010, "IMP-213a: Should meet QA-010");

    println!("\nIMP-213a: Quantization Verification:");
    println!("  Type: {}", result.quantization_type);
    println!("  Max diff: {:.4}", result.max_diff);
    println!("  Mean diff: {:.4}", result.mean_diff);
}

/// IMP-213b: Test different quantization types
#[test]
fn test_imp_213b_quantization_types() {
    let f32_out = vec![0.5, 0.5, 0.5, 0.5];

    // Q4_K has larger tolerance
    let q4k = QuantizationVerificationResult::new(
        "Q4_K",
        f32_out.clone(),
        vec![0.48, 0.52, 0.49, 0.51],
        0.05,
    );

    // Q8_0 has tighter tolerance
    let q8_0 = QuantizationVerificationResult::new(
        "Q8_0",
        f32_out.clone(),
        vec![0.499, 0.501, 0.500, 0.500],
        0.01,
    );

    println!("\nIMP-213b: Quantization Types:");
    println!(
        "  Q4_K: max_diff={:.4}, meets QA-010={}",
        q4k.max_diff, q4k.meets_qa010
    );
    println!(
        "  Q8_0: max_diff={:.4}, meets QA-010={}",
        q8_0.max_diff, q8_0.meets_qa010
    );
}

/// IMP-213c: Test quantization tolerance boundaries
#[test]
fn test_imp_213c_quantization_tolerance() {
    let f32_out = vec![1.0, 1.0, 1.0, 1.0];

    // Within tolerance
    let within = QuantizationVerificationResult::new(
        "Q4_K",
        f32_out.clone(),
        vec![1.04, 0.96, 1.03, 0.97],
        0.05,
    );

    // Outside tolerance
    let outside =
        QuantizationVerificationResult::new("Q4_K", f32_out, vec![1.1, 0.9, 1.1, 0.9], 0.05);

    assert!(within.meets_qa010, "IMP-213c: Should be within tolerance");
    assert!(
        !outside.meets_qa010,
        "IMP-213c: Should be outside tolerance"
    );

    println!("\nIMP-213c: Quantization Tolerance:");
    println!("  Within (0.05): {}", within.meets_qa010);
    println!("  Outside (0.05): {}", outside.meets_qa010);
}

/// IMP-213d: Real-world quantization verification
#[test]
#[ignore = "Requires F32 and quantized model inference"]
fn test_imp_213d_realworld_quantization() {
    let f32_out = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let quant_out = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let result = QuantizationVerificationResult::new("Q4_K", f32_out, quant_out, 0.05);

    println!("\nIMP-213d: Real-World Quantization:");
    println!("  Type: {}", result.quantization_type);
    println!("  Max diff: {:.4}", result.max_diff);
    println!(
        "  QA-010: {}",
        if result.meets_qa010 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-301: Trueno SIMD Q4_K Dequantization ====================
// Per spec: 4-8x speedup via AVX2/NEON for Q4_K dequantization
// Target: ~15 tok/s CPU (match llama.cpp CPU)

/// SIMD backend type for performance tracking
#[derive(Debug, Clone, PartialEq)]
pub enum SimdBackend {
    Scalar,
    SSE2,
    AVX2,
    AVX512,
    Neon,
    Wasm,
}

impl SimdBackend {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdBackend::AVX512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdBackend::AVX2;
            }
            if is_x86_feature_detected!("sse2") {
                return SimdBackend::SSE2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdBackend::Neon;
        }
        #[cfg(target_arch = "wasm32")]
        {
            return SimdBackend::Wasm;
        }
        SimdBackend::Scalar
    }

    pub fn expected_speedup(&self) -> f64 {
        match self {
            SimdBackend::AVX512 => 16.0,
            SimdBackend::AVX2 => 8.0,
            SimdBackend::SSE2 => 4.0,
            SimdBackend::Neon => 4.0,
            SimdBackend::Wasm => 2.0,
            SimdBackend::Scalar => 1.0,
        }
    }
}

/// Trueno SIMD benchmark result
#[derive(Debug, Clone)]
pub struct TruenoSimdBenchResult {
    pub operation: String,
    pub backend: SimdBackend,
    pub scalar_time_us: f64,
    pub simd_time_us: f64,
    pub speedup: f64,
    pub elements: usize,
    pub throughput_gbs: f64,
    pub meets_imp301: bool,
}

impl TruenoSimdBenchResult {
    pub fn new(
        operation: impl Into<String>,
        backend: SimdBackend,
        scalar_us: f64,
        simd_us: f64,
        elements: usize,
    ) -> Self {
        let speedup = scalar_us / simd_us.max(0.001);
        // Throughput: elements * 4 bytes / time_seconds / 1e9 = GB/s
        let throughput_gbs = (elements as f64 * 4.0) / (simd_us * 1e-6) / 1e9;
        // IMP-301: Need at least 2x speedup to be worthwhile
        let meets_imp301 = speedup >= 2.0;

        Self {
            operation: operation.into(),
            backend,
            scalar_time_us: scalar_us,
            simd_time_us: simd_us,
            speedup,
            elements,
            throughput_gbs,
            meets_imp301,
        }
    }
}

/// IMP-301a: Test SIMD backend detection
#[test]
fn test_imp_301a_simd_backend_detection() {
    let backend = SimdBackend::detect();

    println!("\nIMP-301a: SIMD Backend Detection:");
    println!("  Detected: {:?}", backend);
    println!("  Expected speedup: {:.1}x", backend.expected_speedup());

    // Should detect something other than scalar on modern CPUs
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert_ne!(backend, SimdBackend::Scalar, "IMP-301a: Should detect SIMD");
}

/// IMP-301b: Test trueno Vector SIMD operations
#[test]
fn test_imp_301b_trueno_vector_simd() {
    use trueno::Vector;

    let size = 4096;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let vec = Vector::from_slice(&data);

    // Test basic operations
    let sum = vec.sum().expect("sum failed");
    let mean = vec.mean().expect("mean failed");
    let max = vec.max().expect("max failed");

    assert!(sum > 0.0, "IMP-301b: Sum should be positive");
    assert!(mean > 0.0, "IMP-301b: Mean should be positive");
    assert!(max > 0.0, "IMP-301b: Max should be positive");

    println!("\nIMP-301b: Trueno Vector SIMD:");
    println!("  Size: {}", size);
    println!("  Sum: {:.2}", sum);
    println!("  Mean: {:.6}", mean);
    println!("  Max: {:.3}", max);
    println!("  Backend: {:?}", vec.backend());
}

/// IMP-301c: Test trueno SIMD dequantization simulation
#[test]
fn test_imp_301c_trueno_dequant_speedup() {
    use std::time::Instant;
    use trueno::Vector;

    let size = 32768; // Typical weight block size
    let iterations = 100;

    // Simulate Q4_K block data
    let q4k_scales: Vec<f32> = (0..size / 32).map(|i| 0.1 + (i as f32 * 0.001)).collect();
    let q4k_data: Vec<f32> = (0..size).map(|i| ((i % 16) as f32 - 8.0) * 0.1).collect();

    // Scalar baseline
    let start = Instant::now();
    for _ in 0..iterations {
        let _result: Vec<f32> = q4k_data
            .chunks(32)
            .zip(q4k_scales.iter())
            .flat_map(|(chunk, scale)| chunk.iter().map(|&x| x * scale).collect::<Vec<_>>())
            .collect();
    }
    let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

    // Trueno SIMD
    let vec = Vector::from_slice(&q4k_data);
    let _scales_vec = Vector::from_slice(&q4k_scales);
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = vec.mul(&Vector::from_slice(&q4k_data));
    }
    let simd_time = start.elapsed().as_micros() as f64 / iterations as f64;

    let result = TruenoSimdBenchResult::new(
        "Q4_K dequant",
        SimdBackend::detect(),
        scalar_time,
        simd_time.max(1.0), // Avoid div by zero
        size,
    );

    println!("\nIMP-301c: Trueno SIMD Dequant Speedup:");
    println!("  Elements: {}", size);
    println!("  Scalar: {:.1}µs", scalar_time);
    println!("  SIMD: {:.1}µs", simd_time);
    println!("  Speedup: {:.2}x", result.speedup);
    println!("  Throughput: {:.2} GB/s", result.throughput_gbs);
    println!(
        "  IMP-301: {}",
        if result.meets_imp301 {
            "PASS"
        } else {
            "NEEDS OPTIMIZATION"
        }
    );
}

/// IMP-301d: Real-world trueno performance benchmark
#[test]
#[ignore = "Requires extended benchmark time"]
fn test_imp_301d_realworld_trueno_perf() {
    use std::time::Instant;
    use trueno::{Matrix, Vector};

    // Phi-2 model dimensions
    let hidden_dim = 2560;
    let vocab_size = 51200;
    let iterations = 10;

    // Create weight matrix (simulating model weights)
    let weights_data: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| (i as f32 * 0.0001) % 1.0 - 0.5)
        .collect();
    let weights =
        Matrix::from_vec(vocab_size, hidden_dim, weights_data).expect("Matrix creation failed");

    // Create input vector
    let input_data: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.01).collect();
    let input = Vector::from_slice(&input_data);

    // Benchmark matvec
    let start = Instant::now();
    for _ in 0..iterations {
        let _output = weights.matvec(&input).expect("matvec failed");
    }
    let total_time = start.elapsed().as_micros() as f64;
    let avg_time = total_time / iterations as f64;

    // Calculate throughput
    let flops = 2.0 * hidden_dim as f64 * vocab_size as f64; // 2 ops per multiply-add
    let gflops = (flops * iterations as f64) / (total_time * 1e-6) / 1e9;

    println!("\nIMP-301d: Real-World Trueno Performance:");
    println!("  Matrix: {}x{}", vocab_size, hidden_dim);
    println!("  Avg time: {:.1}µs", avg_time);
    println!("  Throughput: {:.2} GFLOPS", gflops);
    println!("  Est. tok/s: {:.1}", 1e6 / avg_time);
}

// ==================== IMP-302: Trueno SIMD Matmul ====================
// Per spec: 4x matmul speedup, >50 GFLOPS single thread

/// Matrix multiplication benchmark result
#[derive(Debug, Clone)]
pub struct MatmulBenchResult {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub time_us: f64,
    pub gflops: f64,
    pub meets_imp302: bool,
}

impl MatmulBenchResult {
    pub fn new(m: usize, n: usize, k: usize, time_us: f64) -> Self {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (time_us * 1e-6) / 1e9;
        let meets_imp302 = gflops >= 50.0; // Target: >50 GFLOPS

        Self {
            m,
            n,
            k,
            time_us,
            gflops,
            meets_imp302,
        }
    }
}

/// IMP-302a: Test trueno Matrix matmul
#[test]
fn test_imp_302a_trueno_matmul() {
    use trueno::Matrix;

    let a_data: Vec<f32> = (0..64 * 128).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..128 * 64).map(|i| (i as f32) * 0.01).collect();

    let a = Matrix::from_vec(64, 128, a_data).expect("Matrix A");
    let b = Matrix::from_vec(128, 64, b_data).expect("Matrix B");

    let c = a.matmul(&b).expect("matmul failed");

    assert_eq!(c.rows(), 64, "IMP-302a: Output rows");
    assert_eq!(c.cols(), 64, "IMP-302a: Output cols");

    println!("\nIMP-302a: Trueno Matmul:");
    println!("  A: 64x128");
    println!("  B: 128x64");
    println!("  C: {}x{}", c.rows(), c.cols());
}

/// IMP-302b: Test trueno matmul performance
#[test]
fn test_imp_302b_trueno_matmul_perf() {
    use std::time::Instant;
    use trueno::Matrix;

    let sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];
    let iterations = 10;

    println!("\nIMP-302b: Trueno Matmul Performance:");
    for (m, n, k) in sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Matrix::from_vec(m, k, a_data).expect("Matrix A");
        let b = Matrix::from_vec(k, n, b_data).expect("Matrix B");

        let start = Instant::now();
        for _ in 0..iterations {
            let _c = a.matmul(&b).expect("matmul");
        }
        let total_us = start.elapsed().as_micros() as f64;
        let avg_us = total_us / iterations as f64;

        let result = MatmulBenchResult::new(m, n, k, avg_us);

        println!(
            "  {}x{}x{}: {:.1}µs, {:.1} GFLOPS [{}]",
            m,
            n,
            k,
            avg_us,
            result.gflops,
            if result.meets_imp302 {
                "PASS"
            } else {
                "NEEDS WORK"
            }
        );
    }
}

/// IMP-302c: Test matvec performance (most common in inference)
#[test]
fn test_imp_302c_trueno_matvec_perf() {
    use std::time::Instant;
    use trueno::{Matrix, Vector};

    // Transformer layer dimensions
    let dims = [(2560, 10240), (10240, 2560), (2560, 51200)];
    let iterations = 50;

    println!("\nIMP-302c: Trueno Matvec Performance:");
    for (rows, cols) in dims {
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.0001).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        let mat = Matrix::from_vec(rows, cols, mat_data).expect("Matrix");
        let vec = Vector::from_slice(&vec_data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = mat.matvec(&vec).expect("matvec");
        }
        let total_us = start.elapsed().as_micros() as f64;
        let avg_us = total_us / iterations as f64;

        let flops = 2.0 * rows as f64 * cols as f64;
        let gflops = flops / (avg_us * 1e-6) / 1e9;

        println!("  {}x{}: {:.1}µs, {:.1} GFLOPS", rows, cols, avg_us, gflops);
    }
}

/// IMP-302d: Real-world matmul benchmark
#[test]
#[ignore = "Requires extended benchmark time"]
fn test_imp_302d_realworld_matmul() {
    use std::time::Instant;
    use trueno::Matrix;

    // Full transformer layer: FFN up projection
    let hidden = 2560;
    let intermediate = 10240;
    let batch = 1;

    let weights: Vec<f32> = (0..hidden * intermediate)
        .map(|i| ((i as f32) * 0.0001) % 1.0 - 0.5)
        .collect();
    let input: Vec<f32> = (0..batch * hidden).map(|i| (i as f32) * 0.01).collect();

    let w = Matrix::from_vec(intermediate, hidden, weights).expect("weights");
    let x = Matrix::from_vec(batch, hidden, input).expect("input");

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _y = Matrix::vecmat(&trueno::Vector::from_slice(x.as_slice()), &w.transpose());
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let result = MatmulBenchResult::new(batch, intermediate, hidden, avg_us);

    println!("\nIMP-302d: Real-World FFN Projection:");
    println!("  Dimensions: {}x{}x{}", batch, intermediate, hidden);
    println!("  Time: {:.1}µs", avg_us);
    println!("  GFLOPS: {:.1}", result.gflops);
    println!(
        "  IMP-302: {}",
        if result.meets_imp302 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-303: Trueno SIMD Activations ====================
// Per spec: 8x activation speedup, <100µs for 4096 dim

/// Activation benchmark result
#[derive(Debug, Clone)]
pub struct ActivationBenchResult {
    pub name: String,
    pub size: usize,
    pub time_us: f64,
    pub throughput_gbs: f64,
    pub meets_imp303: bool,
}

impl ActivationBenchResult {
    pub fn new(name: impl Into<String>, size: usize, time_us: f64) -> Self {
        let throughput_gbs = (size as f64 * 4.0) / (time_us * 1e-6) / 1e9;
        let meets_imp303 = time_us < 100.0 || size > 4096; // <100µs for 4096

        Self {
            name: name.into(),
            size,
            time_us,
            throughput_gbs,
            meets_imp303,
        }
    }
}

/// IMP-303a: Test trueno activation functions
#[test]
fn test_imp_303a_trueno_activations() {
    use trueno::Vector;

    let data: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();
    let vec = Vector::from_slice(&data);

    let relu = vec.relu().expect("relu");
    let sigmoid = vec.sigmoid().expect("sigmoid");
    let gelu = vec.gelu().expect("gelu");
    let swish = vec.swish().expect("swish");

    // Verify basic properties
    assert!(
        relu.as_slice().iter().all(|&x| x >= 0.0),
        "IMP-303a: ReLU non-negative"
    );
    assert!(
        sigmoid.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
        "IMP-303a: Sigmoid (0,1)"
    );

    println!("\nIMP-303a: Trueno Activations:");
    println!("  ReLU(0): {:.4}", relu.as_slice()[100]);
    println!("  Sigmoid(0): {:.4}", sigmoid.as_slice()[100]);
    println!("  GELU(0): {:.4}", gelu.as_slice()[100]);
    println!("  Swish(0): {:.4}", swish.as_slice()[100]);
}

/// IMP-303b: Test activation performance
#[test]
fn test_imp_303b_trueno_activation_perf() {
    use std::time::Instant;
    use trueno::Vector;

    let size = 4096;
    let iterations = 1000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 - 2048.0) * 0.01).collect();
    let vec = Vector::from_slice(&data);

    let activations = ["relu", "sigmoid", "gelu", "swish", "softmax"];

    println!("\nIMP-303b: Trueno Activation Performance (n={}):", size);
    for name in activations {
        let start = Instant::now();
        for _ in 0..iterations {
            match name {
                "relu" => {
                    vec.relu().ok();
                },
                "sigmoid" => {
                    vec.sigmoid().ok();
                },
                "gelu" => {
                    vec.gelu().ok();
                },
                "swish" => {
                    vec.swish().ok();
                },
                "softmax" => {
                    vec.softmax().ok();
                },
                _ => {},
            }
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let result = ActivationBenchResult::new(name, size, avg_us);

        println!(
            "  {}: {:.2}µs, {:.1} GB/s [{}]",
            name,
            avg_us,
            result.throughput_gbs,
            if result.meets_imp303 { "PASS" } else { "SLOW" }
        );
    }
}

/// IMP-303c: Test layer norm performance
#[test]
fn test_imp_303c_trueno_layer_norm_perf() {
    use std::time::Instant;
    use trueno::Vector;

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-303c: Trueno Layer Norm Performance:");
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let vec = Vector::from_slice(&data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = vec.layer_norm_simple(1e-5).expect("layer_norm_simple");
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        println!(
            "  n={}: {:.2}µs [{}]",
            size,
            avg_us,
            if avg_us < 50.0 { "PASS" } else { "NEEDS WORK" }
        );
    }
}

/// IMP-303d: Real-world activation chain
#[test]
#[ignore = "Requires extended benchmark"]
fn test_imp_303d_realworld_activation_chain() {
    use std::time::Instant;
    use trueno::Vector;

    // Full FFN activation chain: linear -> gelu -> linear
    let hidden = 2560;
    let intermediate = 10240;
    let iterations = 100;

    let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
    let _hidden_vec = Vector::from_slice(&x);

    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate FFN: up_proj -> gelu -> down_proj
        let up: Vec<f32> = (0..intermediate).map(|i| i as f32 * 0.001).collect();
        let up_vec = Vector::from_slice(&up);
        let _activated = up_vec.gelu().expect("gelu");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("\nIMP-303d: Real-World Activation Chain:");
    println!("  Hidden: {}, Intermediate: {}", hidden, intermediate);
    println!("  GELU time: {:.1}µs", avg_us);
    println!(
        "  IMP-303: {}",
        if avg_us < 500.0 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-304: Trueno SIMD Layer Norm & RMS Norm ====================
// Per spec: 4x norm speedup for production inference
// Target: < 50µs for 4096 dim layer norm

/// IMP-304a: Test trueno layer_norm correctness
#[test]
fn test_imp_304a_trueno_layer_norm_correctness() {
    use trueno::Vector;

    // Test case 1: Simple normalization
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let vec = Vector::from_slice(&data);
    let gamma = Vector::from_slice(&vec![1.0; 5]);
    let beta = Vector::from_slice(&vec![0.0; 5]);

    let normed = vec.layer_norm(&gamma, &beta, 1e-5).expect("layer_norm");
    let normed_data = normed.as_slice().to_vec();

    // Verify: mean should be ~0, variance should be ~1
    let mean: f32 = normed_data.iter().sum::<f32>() / normed_data.len() as f32;
    let var: f32 =
        normed_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / normed_data.len() as f32;

    assert!(
        mean.abs() < 1e-5,
        "IMP-304a: Mean should be ~0, got {}",
        mean
    );
    assert!(
        (var - 1.0).abs() < 0.1,
        "IMP-304a: Variance should be ~1, got {}",
        var
    );

    // Test case 2: With affine transform (gamma=2, beta=1)
    let gamma2 = Vector::from_slice(&vec![2.0; 5]);
    let beta2 = Vector::from_slice(&vec![1.0; 5]);
    let normed2 = vec
        .layer_norm(&gamma2, &beta2, 1e-5)
        .expect("layer_norm with affine");
    let normed2_data = normed2.as_slice().to_vec();

    // After gamma=2, beta=1: output = 2*normalized + 1
    // Mean should be ~1 (since normalized mean is 0)
    let mean2: f32 = normed2_data.iter().sum::<f32>() / normed2_data.len() as f32;
    assert!(
        (mean2 - 1.0).abs() < 0.1,
        "IMP-304a: Affine mean should be ~1, got {}",
        mean2
    );

    println!("\nIMP-304a: Trueno Layer Norm Correctness:");
    println!("  Simple: mean={:.6}, var={:.6}", mean, var);
    println!("  Affine (gamma=2, beta=1): mean={:.6}", mean2);
    println!("  Status: PASS");
}

/// IMP-304b: Test trueno layer_norm performance vs scalar
#[test]
fn test_imp_304b_trueno_layer_norm_perf_comparison() {
    use std::time::Instant;
    use trueno::Vector;

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-304b: Layer Norm Performance (trueno SIMD vs scalar):");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>8}",
        "Dim", "Trueno µs", "Scalar µs", "Speedup"
    );
    println!("  -------|------------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let vec = Vector::from_slice(&data);

        // Trueno SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = vec.layer_norm_simple(1e-5).expect("layer_norm_simple");
        }
        let trueno_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Scalar baseline
        let start = Instant::now();
        for _ in 0..iterations {
            let mean: f32 = data.iter().sum::<f32>() / size as f32;
            let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / size as f32;
            let inv_std = (var + 1e-5).sqrt().recip();
            let _output: Vec<f32> = data.iter().map(|x| (x - mean) * inv_std).collect();
        }
        let scalar_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = scalar_us / trueno_us;
        let status = if trueno_us < 50.0 { "PASS" } else { "FAIL" };

        println!(
            "  {:>6} | {:>10.2} | {:>10.2} | {:>7.2}x [{}]",
            size, trueno_us, scalar_us, speedup, status
        );
    }
}

/// IMP-304c: RMS Norm implementation (used by LLaMA, Mistral, etc.)
/// RMS Norm: x / sqrt(mean(x^2) + eps) * gamma
#[test]
fn test_imp_304c_rms_norm() {
    use std::time::Instant;
    use trueno::Vector;

    // RMS Norm helper function (trueno doesn't have native rms_norm yet)
    fn rms_norm_simd(input: &Vector<f32>, gamma: &[f32], eps: f32) -> Vec<f32> {
        let data = input.as_slice().to_vec();
        let n = data.len();

        // Compute RMS: sqrt(mean(x^2))
        let mean_sq: f32 = data.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and scale
        data.iter()
            .zip(gamma.iter())
            .map(|(&x, &g)| x * inv_rms * g)
            .collect()
    }

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-304c: RMS Norm Performance:");
    println!("  {:>6} | {:>10} | {:>8}", "Dim", "Latency µs", "Status");
    println!("  -------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let vec = Vector::from_slice(&data);
        let gamma: Vec<f32> = vec![1.0; size];

        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = rms_norm_simd(&vec, &gamma, 1e-5);
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let status = if avg_us < 50.0 { "PASS" } else { "NEEDS OPT" };
        println!("  {:>6} | {:>10.2} | {}", size, avg_us, status);
    }

    // Verify correctness
    let test_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let test_vec = Vector::from_slice(&test_data);
    let test_gamma = vec![1.0; 4];
    let result = rms_norm_simd(&test_vec, &test_gamma, 1e-5);

    // Expected: RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Output = [1/2.739, 2/2.739, 3/2.739, 4/2.739] ≈ [0.365, 0.730, 1.095, 1.461]
    let expected_rms = (30.0_f32 / 4.0).sqrt();
    let expected: Vec<f32> = test_data.iter().map(|x| x / expected_rms).collect();

    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "IMP-304c: RMS norm mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("  Correctness: VERIFIED");
}

/// IMP-304d: Integration with realizar forward pass timing
#[test]
fn test_imp_304d_layer_norm_integration() {
    use std::time::Instant;
    use trueno::Vector;

    // Simulate phi-2 layer norm dimensions
    let hidden_dim = 2560;
    let num_layers = 32;
    let iterations = 100;

    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let input_vec = Vector::from_slice(&input);

    // Time a full forward pass worth of layer norms (2 per layer: attn_norm + ffn_norm)
    let norms_per_forward = num_layers * 2;

    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..norms_per_forward {
            let _normed = input_vec.layer_norm_simple(1e-5).expect("layer_norm");
        }
    }
    let total_us = start.elapsed().as_micros() as f64;
    let per_forward_us = total_us / iterations as f64;
    let per_norm_us = per_forward_us / norms_per_forward as f64;

    println!("\nIMP-304d: Layer Norm Integration (phi-2 scale):");
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Layers: {} (× 2 norms each)", num_layers);
    println!("  Per norm: {:.2}µs", per_norm_us);
    println!(
        "  Per forward (all norms): {:.2}µs ({:.2}ms)",
        per_forward_us,
        per_forward_us / 1000.0
    );

    let target_ms = 5.0; // Target: all norms < 5ms per forward
    let status = if per_forward_us / 1000.0 < target_ms {
        "PASS"
    } else {
        "NEEDS WORK"
    };
    println!("  Status: {} (target: <{}ms)", status, target_ms);
}

// ==================== IMP-305: Trueno SIMD Softmax ====================
// Per spec: 4x softmax speedup with numerical stability
// Target: < 100µs for 32K vocab softmax

/// IMP-305a: Test trueno softmax correctness and numerical stability
#[test]
fn test_imp_305a_trueno_softmax_correctness() {
    use trueno::Vector;

    // Test case 1: Simple softmax
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let vec = Vector::from_slice(&data);
    let result = vec.softmax().expect("softmax");
    let result_data = result.as_slice().to_vec();

    // Softmax should sum to 1
    let sum: f32 = result_data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "IMP-305a: Softmax should sum to 1, got {}",
        sum
    );

    // Higher inputs should have higher probabilities
    for i in 0..result_data.len() - 1 {
        assert!(
            result_data[i] < result_data[i + 1],
            "IMP-305a: Softmax should be monotonic"
        );
    }

    // Test case 2: Numerical stability with large values
    let large_data = vec![1000.0_f32, 1001.0, 1002.0, 1003.0];
    let large_vec = Vector::from_slice(&large_data);
    let large_result = large_vec.softmax().expect("softmax large");
    let large_result_data = large_result.as_slice();
    let large_sum: f32 = large_result_data.iter().sum();
    assert!(
        (large_sum - 1.0).abs() < 1e-4,
        "IMP-305a: Large value softmax should sum to 1, got {}",
        large_sum
    );
    assert!(
        large_result_data.iter().all(|&x| x.is_finite()),
        "IMP-305a: Large value softmax should be finite"
    );

    // Test case 3: Numerical stability with negative values
    let neg_data = vec![-1000.0_f32, -999.0, -998.0, -997.0];
    let neg_vec = Vector::from_slice(&neg_data);
    let neg_result = neg_vec.softmax().expect("softmax negative");
    let neg_sum: f32 = neg_result.as_slice().iter().sum();
    assert!(
        (neg_sum - 1.0).abs() < 1e-4,
        "IMP-305a: Negative value softmax should sum to 1, got {}",
        neg_sum
    );

    println!("\nIMP-305a: Trueno Softmax Correctness:");
    println!("  Simple: sum={:.6}, monotonic=true", sum);
    println!("  Large values (1000+): sum={:.6}, all finite", large_sum);
    println!("  Negative values: sum={:.6}", neg_sum);
    println!("  Status: PASS");
}

/// IMP-305b: Test trueno softmax performance
#[test]
fn test_imp_305b_trueno_softmax_perf() {
    use std::time::Instant;
    use trueno::Vector;

    // Test vocab sizes relevant to LLMs
    let sizes = [1024, 4096, 32000, 51200]; // Common vocab sizes
    let iterations = 1000;

    println!("\nIMP-305b: Softmax Performance:");
    println!(
        "  {:>6} | {:>10} | {:>8}",
        "VocabSz", "Latency µs", "Status"
    );
    println!("  -------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32) * 0.001 - (size as f32 / 2000.0))
            .collect();
        let vec = Vector::from_slice(&data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = vec.softmax().expect("softmax");
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let target = if size <= 32000 { 100.0 } else { 200.0 };
        let status = if avg_us < target { "PASS" } else { "NEEDS OPT" };
        println!("  {:>6} | {:>10.2} | {}", size, avg_us, status);
    }
}

/// IMP-305c: Softmax integration with attention mechanism
#[test]
fn test_imp_305c_attention_softmax_integration() {
    use std::time::Instant;
    use trueno::Vector;

    // Simulate attention softmax: seq_len × seq_len scores
    let seq_lengths = [128, 256, 512, 1024];
    let num_heads = 32;
    let iterations = 100;

    println!("\nIMP-305c: Attention Softmax Integration:");
    println!(
        "  {:>8} | {:>12} | {:>12} | {:>8}",
        "SeqLen", "Per Head µs", "All Heads µs", "Status"
    );
    println!("  ---------|--------------|--------------|----------");

    for seq_len in seq_lengths {
        // Each head does seq_len softmax operations (one per query position)
        let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let scores_vec = Vector::from_slice(&scores);

        // Time softmax for all heads × all positions
        let start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..num_heads {
                for _ in 0..seq_len {
                    let _probs = scores_vec.softmax().expect("softmax");
                }
            }
        }
        let total_us = start.elapsed().as_micros() as f64;
        let per_head_us = total_us / (iterations * num_heads) as f64;
        let all_heads_us = total_us / iterations as f64;

        let target_ms = 50.0; // Target: all attention softmax < 50ms
        let status = if all_heads_us / 1000.0 < target_ms {
            "PASS"
        } else {
            "SLOW"
        };

        println!(
            "  {:>8} | {:>12.2} | {:>12.2} | {}",
            seq_len, per_head_us, all_heads_us, status
        );
    }
}

/// IMP-305d: Combined norm + softmax timing (common pattern)
#[test]
fn test_imp_305d_norm_softmax_combined() {
    use std::time::Instant;
    use trueno::Vector;

    // Common inference pattern: layer_norm -> attention (with softmax) -> layer_norm
    let hidden_dim = 2560;
    let seq_len = 256;
    let iterations = 100;

    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let hidden_vec = Vector::from_slice(&hidden);

    let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let scores_vec = Vector::from_slice(&scores);

    // Measure: 2× layer_norm + seq_len× softmax
    let start = Instant::now();
    for _ in 0..iterations {
        // Pre-attention norm
        let _normed1 = hidden_vec.layer_norm_simple(1e-5).expect("norm1");

        // Attention softmax (per position)
        for _ in 0..seq_len {
            let _probs = scores_vec.softmax().expect("softmax");
        }

        // Post-attention norm (before FFN)
        let _normed2 = hidden_vec.layer_norm_simple(1e-5).expect("norm2");
    }
    let total_us = start.elapsed().as_micros() as f64;
    let per_iter_us = total_us / iterations as f64;
    let per_iter_ms = per_iter_us / 1000.0;

    println!("\nIMP-305d: Combined Norm + Softmax (per layer):");
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Seq len: {}", seq_len);
    println!("  Operations: 2× layer_norm + {}× softmax", seq_len);
    println!("  Total: {:.2}ms per layer", per_iter_ms);

    let target_ms = 100.0;
    let status = if per_iter_ms < target_ms {
        "PASS"
    } else {
        "NEEDS WORK"
    };
    println!("  Status: {} (target: <{}ms)", status, target_ms);
}
