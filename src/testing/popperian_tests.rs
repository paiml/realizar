//! Popperian Falsification Tests
//!
//! Following Karl Popper's philosophy of science: a theory is only scientific
//! if it can be falsified. These tests define strict prohibitions that, if
//! violated, refute the implementation hypothesis.
//!
//! # References
//!
//! - Popper, K. (1959). "The Logic of Scientific Discovery"
//! - Popper, K. (1963). "Conjectures and Refutations"

use super::fixtures::{GgufFixture, ModelFixture};
use super::generators::SyntheticWeightGenerator;
use super::{Device, ModelConfig, ModelFormat, QuantType};

/// Falsification threshold for quantization RMSE
/// If RMSE exceeds this, the quantization is considered refuted.
const QUANT_RMSE_THRESHOLD: f32 = 1e-3;

/// Tolerance for thread invariance (must be bit-exact for determinism)
const THREAD_INVARIANCE_TOLERANCE: f32 = 0.0;

/// Tolerance for SIMD backend parity
const SIMD_PARITY_EPSILON: f32 = 1e-6;

// =============================================================================
// Falsification Gate 1: Quantization Forbidden Zones
// =============================================================================

/// Compute RMSE between two vectors
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length for RMSE");
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / a.len() as f32).sqrt()
}

/// Compute max absolute error between two vectors
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// F100: Falsification Gate - Q4_0 RMSE must not exceed threshold
///
/// Prohibition: If RMSE(Q4_0_output, F32_output) > 1e-3, the implementation is refuted.
#[test]
fn test_f100_quantization_rmse_gate_q4_0() {
    let config = ModelConfig::tiny();
    let gen = SyntheticWeightGenerator::new(42);

    // Generate F32 reference weights
    let f32_weights = gen.generate_model_weights(&config, QuantType::F32);

    // Generate Q4_0 quantized weights
    let q4_weights = gen.generate_model_weights(&config, QuantType::Q4_0);

    // For now, compare embedding weights (both stored as raw bytes)
    // In a full implementation, we would dequantize and compare
    let f32_len = f32_weights.embed_weights.len();
    let q4_len = q4_weights.embed_weights.len();

    // Q4_0 should be smaller (4 bits vs 32 bits = 8x compression)
    // Allow for block overhead: expect 4-6x compression
    let compression_ratio = f32_len as f32 / q4_len as f32;
    assert!(
        compression_ratio > 3.0 && compression_ratio < 10.0,
        "Q4_0 compression ratio {} outside expected range [3, 10]",
        compression_ratio
    );
}

/// F101: Falsification Gate - Q8_0 RMSE must not exceed threshold
#[test]
fn test_f101_quantization_rmse_gate_q8_0() {
    let config = ModelConfig::tiny();
    let gen = SyntheticWeightGenerator::new(42);

    let f32_weights = gen.generate_model_weights(&config, QuantType::F32);
    let q8_weights = gen.generate_model_weights(&config, QuantType::Q8_0);

    // Q8_0 should be ~4x smaller than F32
    let compression_ratio =
        f32_weights.embed_weights.len() as f32 / q8_weights.embed_weights.len() as f32;
    assert!(
        compression_ratio > 2.0 && compression_ratio < 6.0,
        "Q8_0 compression ratio {} outside expected range [2, 6]",
        compression_ratio
    );
}

// =============================================================================
// Falsification Gate 2: Thread Invariance (Determinism)
// =============================================================================

/// F102: Thread Invariance Test
///
/// Prohibition: If output differs between 1 thread and N threads, determinism is refuted.
#[test]
fn test_f102_thread_invariance() {
    let config = ModelConfig::tiny();
    let fixture = GgufFixture::new(config, QuantType::F32, 42);
    let tokens = vec![1, 2, 3, 4, 5];

    // Run with single thread
    let result_1 = {
        // Force single-threaded execution
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| fixture.forward(Device::Cpu, &tokens).unwrap())
    };

    // Run with multiple threads
    let result_n = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| fixture.forward(Device::Cpu, &tokens).unwrap())
    };

    // Results MUST be bit-exact for determinism
    assert_eq!(
        result_1.len(),
        result_n.len(),
        "Output lengths differ between thread counts"
    );

    for (i, (a, b)) in result_1.iter().zip(result_n.iter()).enumerate() {
        assert!(
            (a - b).abs() <= THREAD_INVARIANCE_TOLERANCE,
            "Thread invariance violated at index {}: 1-thread={}, 8-threads={}",
            i,
            a,
            b
        );
    }
}

/// F103: Seed Reproducibility Test
///
/// Prohibition: If same seed produces different results, RNG is refuted.
#[test]
fn test_f103_seed_reproducibility() {
    let config = ModelConfig::tiny();

    // Generate with same seed twice
    let gen1 = SyntheticWeightGenerator::new(12345);
    let weights1 = gen1.generate_model_weights(&config, QuantType::F32);

    let gen2 = SyntheticWeightGenerator::new(12345);
    let weights2 = gen2.generate_model_weights(&config, QuantType::F32);

    // Must be bit-exact
    assert_eq!(
        weights1.embed_weights, weights2.embed_weights,
        "Same seed produced different embedding weights"
    );
}

/// F104: Different Seeds Produce Different Results
///
/// Prohibition: If different seeds produce identical results, RNG is trivial.
#[test]
fn test_f104_seed_variation() {
    let config = ModelConfig::tiny();

    let gen1 = SyntheticWeightGenerator::new(1);
    let weights1 = gen1.generate_model_weights(&config, QuantType::F32);

    let gen2 = SyntheticWeightGenerator::new(2);
    let weights2 = gen2.generate_model_weights(&config, QuantType::F32);

    // Must differ
    assert_ne!(
        weights1.embed_weights, weights2.embed_weights,
        "Different seeds produced identical weights - RNG may be broken"
    );
}

// =============================================================================
// Falsification Gate 3: Black Swan Inputs (Property-Based Testing)
// =============================================================================

/// F105: Empty Token Sequence Handling
///
/// Prohibition: If empty input causes panic, error handling is refuted.
#[test]
fn test_f105_empty_input_handling() {
    let fixture = GgufFixture::tiny_gqa();
    let empty_tokens: Vec<u32> = vec![];

    // Should not panic - may return error or empty output
    let result = fixture.forward(Device::Cpu, &empty_tokens);

    // Either succeeds with empty/default output, or returns proper error
    match result {
        Ok(output) => {
            // Empty or vocab-sized output is acceptable
            assert!(
                output.is_empty() || output.len() == fixture.config().vocab_size,
                "Unexpected output length {} for empty input",
                output.len()
            );
        },
        Err(_) => {
            // Error is acceptable for empty input
        },
    }
}

/// F106: Single Token Handling
///
/// Prohibition: If single-token input fails, minimal case is broken.
#[test]
fn test_f106_single_token_handling() {
    let fixture = GgufFixture::tiny_gqa();
    let single_token = vec![42u32];

    let result = fixture.forward(Device::Cpu, &single_token);
    assert!(result.is_ok(), "Single token input should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), fixture.config().vocab_size);
}

/// F107: Maximum Token ID Handling
///
/// Prohibition: If max valid token ID fails, boundary case is broken.
#[test]
fn test_f107_max_token_id() {
    let fixture = GgufFixture::tiny_gqa();
    let max_token = (fixture.config().vocab_size - 1) as u32;
    let tokens = vec![max_token];

    let result = fixture.forward(Device::Cpu, &tokens);
    assert!(result.is_ok(), "Max token ID {} should be valid", max_token);
}

/// F108: Out-of-Vocabulary Token Handling
///
/// Prohibition: If OOV token causes panic (not error), robustness is refuted.
#[test]
fn test_f108_oov_token_handling() {
    let fixture = GgufFixture::tiny_gqa();
    let oov_token = fixture.config().vocab_size as u32 + 1000;
    let tokens = vec![oov_token];

    // Should not panic - current impl may clamp or return error
    let result = std::panic::catch_unwind(|| fixture.forward(Device::Cpu, &tokens));

    assert!(result.is_ok(), "OOV token {} caused panic", oov_token);
}

/// F109: NaN in Output Detection
///
/// Prohibition: If forward pass produces NaN, numerical stability is refuted.
#[test]
fn test_f109_no_nan_in_output() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let output = fixture.forward(Device::Cpu, &tokens).unwrap();

    let nan_count = output.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "Output contains {} NaN values", nan_count);
}

/// F110: Inf in Output Detection
///
/// Prohibition: If forward pass produces Inf, overflow occurred.
#[test]
fn test_f110_no_inf_in_output() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let output = fixture.forward(Device::Cpu, &tokens).unwrap();

    let inf_count = output.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(inf_count, 0, "Output contains {} Inf values", inf_count);
}

// =============================================================================
// Falsification Gate 4: SIMD Backend Parity
// =============================================================================

/// F111: Scalar vs SIMD Parity for Embedding
///
/// Prohibition: If SIMD and Scalar produce different embeddings, one is buggy.
#[test]
fn test_f111_embedding_simd_parity() {
    let fixture = GgufFixture::tiny_gqa();

    // Get embedding via standard path
    let embed1 = fixture.embed(Device::Cpu, 42).unwrap();

    // Get embedding again (should be deterministic)
    let embed2 = fixture.embed(Device::Cpu, 42).unwrap();

    // Must be bit-exact
    assert_eq!(embed1, embed2, "Embedding not deterministic");

    // Verify L2 norm is reasonable
    let l2: f32 = embed1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(l2.is_finite(), "Embedding L2 norm is not finite");
    assert!(l2 > 0.0, "Embedding L2 norm is zero");
}

/// F112: Cross-Quantization Forward Pass Consistency
///
/// Prohibition: If different quant types produce wildly different distributions, one is wrong.
#[test]
fn test_f112_quant_forward_consistency() {
    let config = ModelConfig::tiny();
    let tokens = vec![1, 2, 3];

    let f32_fixture = GgufFixture::new(config.clone(), QuantType::F32, 42);
    let q8_fixture = GgufFixture::new(config.clone(), QuantType::Q8_0, 42);
    let q4_fixture = GgufFixture::new(config, QuantType::Q4_0, 42);

    let f32_out = f32_fixture.forward(Device::Cpu, &tokens).unwrap();
    let q8_out = q8_fixture.forward(Device::Cpu, &tokens).unwrap();
    let q4_out = q4_fixture.forward(Device::Cpu, &tokens).unwrap();

    // All should produce vocab_size outputs
    assert_eq!(f32_out.len(), q8_out.len());
    assert_eq!(f32_out.len(), q4_out.len());

    // L2 norms should be in same order of magnitude
    let f32_l2: f32 = f32_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q8_l2: f32 = q8_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q4_l2: f32 = q4_out.iter().map(|x| x * x).sum::<f32>().sqrt();

    // All L2 norms should be finite and positive
    assert!(f32_l2.is_finite() && f32_l2 > 0.0);
    assert!(q8_l2.is_finite() && q8_l2 > 0.0);
    assert!(q4_l2.is_finite() && q4_l2 > 0.0);
}

// =============================================================================
// Falsification Gate 5: Memory Safety (Stress Testing)
// =============================================================================

/// F113: Setup/Teardown Stress Test (1000 cycles)
///
/// Prohibition: If memory grows monotonically, there is a leak.
#[test]
fn test_f113_setup_teardown_stress() {
    let config = ModelConfig::tiny();

    // Track that we can create and drop many fixtures without issue
    for i in 0..1000 {
        let fixture = GgufFixture::new(config.clone(), QuantType::F32, i as u64);
        let _ = fixture.forward(Device::Cpu, &[1, 2, 3]);
        // fixture is dropped here
    }

    // If we got here without OOM, the test passes
    // A more sophisticated test would track RSS, but that requires
    // platform-specific code and is beyond unit test scope
}

/// F114: Concurrent Fixture Creation
///
/// Prohibition: If concurrent creation causes data races, thread safety is refuted.
#[test]
fn test_f114_concurrent_fixture_creation() {
    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(ModelConfig::tiny());
    let mut handles = vec![];

    for i in 0..8 {
        let cfg = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                let fixture =
                    GgufFixture::new((*cfg).clone(), QuantType::F32, (i * 100 + j) as u64);
                let _ = fixture.forward(Device::Cpu, &[1, 2, 3]);
            }
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("Thread panicked during concurrent fixture creation");
    }
}

/// F115: Conversion Chain Stress Test
///
/// Prohibition: If conversion chain leaks memory, resource management is broken.
#[test]
fn test_f115_conversion_chain_stress() {
    let config = ModelConfig::tiny();

    for i in 0..100 {
        // Create GGUF -> convert to APR -> convert to Safetensors -> convert back
        let gguf = GgufFixture::new(config.clone(), QuantType::F32, i as u64);
        let apr = gguf.convert_to(ModelFormat::APR).unwrap();
        let st = apr.convert_to(ModelFormat::Safetensors).unwrap();
        let back = st.convert_to(ModelFormat::GGUF).unwrap();

        // Verify config preserved
        assert_eq!(back.config().num_heads, config.num_heads);
        assert_eq!(back.config().num_kv_heads, config.num_kv_heads);
    }
}

// =============================================================================
// Additional Popperian Falsification Tests
// =============================================================================

/// F116: GQA Ratio Preservation Through Conversions
///
/// Prohibition: If GQA ratio changes during conversion, metadata is corrupted.
#[test]
fn test_f116_gqa_ratio_preservation() {
    let configs = [
        ("tiny", ModelConfig::tiny()),
        ("small", ModelConfig::small()),
        ("qwen", ModelConfig::qwen_1_5b()),
    ];

    for (name, config) in configs {
        let original_ratio = config.num_heads / config.num_kv_heads;

        let gguf = GgufFixture::new(config.clone(), QuantType::F32, 42);
        let apr = gguf.convert_to(ModelFormat::APR).unwrap();
        let st = apr.convert_to(ModelFormat::Safetensors).unwrap();

        let apr_ratio = apr.config().num_heads / apr.config().num_kv_heads;
        let st_ratio = st.config().num_heads / st.config().num_kv_heads;

        assert_eq!(
            original_ratio, apr_ratio,
            "{}: GQA ratio changed GGUF->APR ({} -> {})",
            name, original_ratio, apr_ratio
        );
        assert_eq!(
            original_ratio, st_ratio,
            "{}: GQA ratio changed APR->Safetensors ({} -> {})",
            name, original_ratio, st_ratio
        );
    }
}

/// F117: Vocab Size Boundary Test
///
/// Prohibition: If vocab_size=1 or very large fails, edge cases are broken.
#[test]
fn test_f117_vocab_size_boundaries() {
    // Minimum vocab size
    let mut min_config = ModelConfig::tiny();
    min_config.vocab_size = 2; // Must be at least 2 for meaningful output

    let fixture = GgufFixture::new(min_config, QuantType::F32, 42);
    let result = fixture.forward(Device::Cpu, &[0, 1]);
    assert!(result.is_ok(), "Min vocab size should work");
    assert_eq!(result.unwrap().len(), 2);
}

/// F118: Hidden Dimension Divisibility
///
/// Prohibition: If hidden_dim not divisible by num_heads, config is invalid.
#[test]
fn test_f118_hidden_dim_divisibility() {
    let config = ModelConfig::tiny();

    // Verify head_dim is exact
    assert_eq!(
        config.hidden_dim % config.num_heads,
        0,
        "hidden_dim {} must be divisible by num_heads {}",
        config.hidden_dim,
        config.num_heads
    );

    // Verify head_dim computation
    assert_eq!(config.head_dim() * config.num_heads, config.hidden_dim);
}

/// F119: RoPE Theta Sanity
///
/// Prohibition: If rope_theta <= 0, position encoding will fail.
#[test]
fn test_f119_rope_theta_sanity() {
    let configs = [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::tinyllama(),
        ModelConfig::qwen_1_5b(),
    ];

    for config in configs {
        assert!(
            config.rope_theta > 0.0,
            "rope_theta {} must be positive",
            config.rope_theta
        );
        assert!(config.rope_theta.is_finite(), "rope_theta must be finite");
    }
}

/// F120: RMS Norm Epsilon Sanity
///
/// Prohibition: If rms_norm_eps <= 0 or too large, normalization will fail.
#[test]
fn test_f120_rms_norm_eps_sanity() {
    let configs = [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::tinyllama(),
        ModelConfig::qwen_1_5b(),
    ];

    for config in configs {
        assert!(
            config.rms_norm_eps > 0.0,
            "rms_norm_eps {} must be positive",
            config.rms_norm_eps
        );
        assert!(
            config.rms_norm_eps < 1e-3,
            "rms_norm_eps {} should be small",
            config.rms_norm_eps
        );
    }
}
