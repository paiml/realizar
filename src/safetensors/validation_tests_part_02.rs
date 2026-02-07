//! T-COV-95 Coverage Bridge: safetensors/validation.rs Part 02
//!
//! Targets uncovered lines: ValidatedWeight accessors (data, into_inner, out_dim, in_dim, name, stats),
//! ValidatedVector accessors (data, into_inner, name, stats),
//! ValidatedEmbedding accessors (data, into_inner, vocab_size, hidden_dim, stats),
//! ContractValidationError Display/Error/From impls,
//! validate_weight edge cases, validate_vector edge cases,
//! enforce_embedding_validation, enforce_weight_validation,
//! ValidatedAprTransformer with optional biases,
//! TensorStats edge cases (zero_pct, empty data).

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::RealizarError;
use crate::safetensors::validation::*;

// ============================================================================
// TensorStats edge cases
// ============================================================================

#[test]
fn test_tensor_stats_empty_data() {
    let stats = TensorStats::compute(&[]);
    assert_eq!(stats.len, 0);
    assert_eq!(stats.zero_count, 0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.l2_norm, 0.0);
}

#[test]
fn test_tensor_stats_zero_pct_empty() {
    let stats = TensorStats::compute(&[]);
    assert_eq!(stats.zero_pct(), 0.0);
}

#[test]
fn test_tensor_stats_zero_pct_all_zeros() {
    let data = vec![0.0f32; 100];
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 100.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_zero_pct_half_zeros() {
    let mut data = vec![0.0f32; 50];
    data.extend(vec![1.0f32; 50]);
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 50.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_zero_pct_no_zeros() {
    let data = vec![1.0f32; 100];
    let stats = TensorStats::compute(&data);
    assert!((stats.zero_pct() - 0.0).abs() < 0.01);
}

#[test]
fn test_tensor_stats_with_nan() {
    let data = vec![f32::NAN, 1.0, 2.0, f32::NAN];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.nan_count, 2);
    assert_eq!(stats.len, 4);
}

#[test]
fn test_tensor_stats_with_inf() {
    let data = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.inf_count, 2);
}

#[test]
fn test_tensor_stats_all_nan() {
    let data = vec![f32::NAN; 10];
    let stats = TensorStats::compute(&data);
    assert_eq!(stats.nan_count, 10);
    // min/max should be 0.0 since all values were NaN
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
}

#[test]
fn test_tensor_stats_single_element() {
    let stats = TensorStats::compute(&[42.0]);
    assert_eq!(stats.len, 1);
    assert_eq!(stats.zero_count, 0);
    assert!((stats.min - 42.0).abs() < 0.001);
    assert!((stats.max - 42.0).abs() < 0.001);
    assert!((stats.mean - 42.0).abs() < 0.001);
}

#[test]
fn test_tensor_stats_negative_values() {
    let data = vec![-1.0, -2.0, -3.0, 1.0, 2.0, 3.0];
    let stats = TensorStats::compute(&data);
    assert!((stats.min - (-3.0)).abs() < 0.001);
    assert!((stats.max - 3.0).abs() < 0.001);
    assert!(stats.mean.abs() < 0.001); // mean should be ~0
}

#[test]
fn test_tensor_stats_debug() {
    let stats = TensorStats::compute(&[1.0, 2.0, 3.0]);
    let debug = format!("{:?}", stats);
    assert!(debug.contains("TensorStats"));
}

#[test]
fn test_tensor_stats_clone() {
    let stats = TensorStats::compute(&[1.0, 2.0, 3.0]);
    let cloned = stats.clone();
    assert_eq!(cloned.len, stats.len);
    assert!((cloned.mean - stats.mean).abs() < 0.001);
}

// ============================================================================
// ValidatedEmbedding accessor tests
// ============================================================================

#[test]
fn test_validated_embedding_data_accessor() {
    let vocab_size = 10;
    let hidden_dim = 8;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve =
        ValidatedEmbedding::new(data.clone(), vocab_size, hidden_dim).expect("should validate");

    assert_eq!(ve.data().len(), vocab_size * hidden_dim);
    assert!((ve.data()[0] - data[0]).abs() < 1e-6);
}

#[test]
fn test_validated_embedding_vocab_size() {
    let vocab_size = 20;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    assert_eq!(ve.vocab_size(), 20);
}

#[test]
fn test_validated_embedding_hidden_dim() {
    let vocab_size = 10;
    let hidden_dim = 16;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    assert_eq!(ve.hidden_dim(), 16);
}

#[test]
fn test_validated_embedding_stats() {
    let vocab_size = 10;
    let hidden_dim = 8;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let stats = ve.stats();
    assert_eq!(stats.len, vocab_size * hidden_dim);
    assert_eq!(stats.nan_count, 0);
}

#[test]
fn test_validated_embedding_into_inner() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let original_len = data.len();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let inner = ve.into_inner();
    assert_eq!(inner.len(), original_len);
}

#[test]
fn test_validated_embedding_debug() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let debug = format!("{:?}", ve);
    assert!(debug.contains("ValidatedEmbedding"));
}

#[test]
fn test_validated_embedding_clone() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    let ve = ValidatedEmbedding::new(data, vocab_size, hidden_dim).expect("should validate");
    let cloned = ve.clone();
    assert_eq!(cloned.vocab_size(), ve.vocab_size());
    assert_eq!(cloned.hidden_dim(), ve.hidden_dim());
}

// ============================================================================
// ValidatedEmbedding error paths
// ============================================================================

#[test]
fn test_validated_embedding_rejects_inf() {
    let vocab_size = 5;
    let hidden_dim = 4;
    let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin() * 0.1 + 0.05)
        .collect();
    data[3] = f32::INFINITY;
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_embedding_rejects_zero_l2() {
    // All values extremely close to zero (below 1e-10 threshold)
    let vocab_size = 5;
    let hidden_dim = 4;
    let data = vec![1e-11f32; vocab_size * hidden_dim];
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
}

#[test]
fn test_validated_embedding_rejects_constant() {
    // All identical non-zero values
    let vocab_size = 5;
    let hidden_dim = 4;
    let data = vec![0.5f32; vocab_size * hidden_dim];
    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.rule_id, "F-DATA-QUALITY-003");
}

// ============================================================================
// ValidatedWeight accessor tests
// ============================================================================

#[test]
fn test_validated_weight_data_accessor() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data.clone(), 10, 10, "test_weight").expect("should validate");
    assert_eq!(vw.data().len(), 100);
    assert!((vw.data()[0] - data[0]).abs() < 1e-6);
}

#[test]
fn test_validated_weight_into_inner() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "test_weight").expect("should validate");
    let inner = vw.into_inner();
    assert_eq!(inner.len(), 100);
}

#[test]
fn test_validated_weight_out_dim() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 20, 10, "test").expect("should validate");
    assert_eq!(vw.out_dim(), 20);
}

#[test]
fn test_validated_weight_in_dim() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 20, 10, "test").expect("should validate");
    assert_eq!(vw.in_dim(), 10);
}

#[test]
fn test_validated_weight_name() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "my_weight").expect("should validate");
    assert_eq!(vw.name(), "my_weight");
}

#[test]
fn test_validated_weight_stats() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "test").expect("should validate");
    let stats = vw.stats();
    assert_eq!(stats.len, 100);
    assert_eq!(stats.nan_count, 0);
    assert!(stats.l2_norm > 0.0);
}

#[test]
fn test_validated_weight_debug() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "dbg").expect("should validate");
    let debug = format!("{:?}", vw);
    assert!(debug.contains("ValidatedWeight"));
}

#[test]
fn test_validated_weight_clone() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let vw = ValidatedWeight::new(data, 10, 10, "clone_test").expect("should validate");
    let cloned = vw.clone();
    assert_eq!(cloned.name(), "clone_test");
    assert_eq!(cloned.out_dim(), 10);
}

// ============================================================================
// ValidatedWeight error paths
// ============================================================================

#[test]
fn test_validated_weight_rejects_nan() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::NAN;
    let result = ValidatedWeight::new(data, 10, 10, "nan_weight");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_weight_rejects_inf() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::INFINITY;
    let result = ValidatedWeight::new(data, 10, 10, "inf_weight");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_weight_rejects_zero_l2() {
    let data = vec![1e-11f32; 100]; // All near zero
    let result = ValidatedWeight::new(data, 10, 10, "zero_l2");
    assert!(result.is_err());
}

#[test]
fn test_validated_weight_shape_mismatch() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let result = ValidatedWeight::new(data, 5, 5, "wrong_shape"); // expects 25
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-001");
}

// ============================================================================
// ValidatedVector accessor tests
// ============================================================================

#[test]
fn test_validated_vector_data_accessor() {
    let data = vec![1.0f32; 64];
    let vv = ValidatedVector::new(data, 64, "test_vec").expect("should validate");
    assert_eq!(vv.data().len(), 64);
    assert!((vv.data()[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_validated_vector_into_inner() {
    let data = vec![2.0f32; 32];
    let vv = ValidatedVector::new(data, 32, "test_vec").expect("should validate");
    let inner = vv.into_inner();
    assert_eq!(inner.len(), 32);
    assert!((inner[0] - 2.0).abs() < 0.001);
}

#[test]
fn test_validated_vector_name() {
    let data = vec![1.0f32; 16];
    let vv = ValidatedVector::new(data, 16, "my_bias").expect("should validate");
    assert_eq!(vv.name(), "my_bias");
}

#[test]
fn test_validated_vector_stats() {
    let data = vec![1.0f32; 64];
    let vv = ValidatedVector::new(data, 64, "norm_weight").expect("should validate");
    let stats = vv.stats();
    assert_eq!(stats.len, 64);
    assert_eq!(stats.nan_count, 0);
}

#[test]
fn test_validated_vector_debug() {
    let data = vec![1.0f32; 8];
    let vv = ValidatedVector::new(data, 8, "dbg").expect("should validate");
    let debug = format!("{:?}", vv);
    assert!(debug.contains("ValidatedVector"));
}

#[test]
fn test_validated_vector_clone() {
    let data = vec![1.0f32; 16];
    let vv = ValidatedVector::new(data, 16, "clone_vec").expect("should validate");
    let cloned = vv.clone();
    assert_eq!(cloned.name(), "clone_vec");
    assert_eq!(cloned.data().len(), 16);
}

// ============================================================================
// ValidatedVector error paths
// ============================================================================

#[test]
fn test_validated_vector_rejects_nan() {
    let mut data = vec![1.0f32; 16];
    data[8] = f32::NAN;
    let result = ValidatedVector::new(data, 16, "nan_vec");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_vector_rejects_inf() {
    let mut data = vec![1.0f32; 16];
    data[8] = f32::NEG_INFINITY;
    let result = ValidatedVector::new(data, 16, "inf_vec");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_validated_vector_length_mismatch() {
    let data = vec![1.0f32; 16];
    let result = ValidatedVector::new(data, 32, "wrong_len");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-003");
}

// ============================================================================
// ContractValidationError Display/Error/From impls
// ============================================================================

#[test]
fn test_contract_validation_error_display() {
    let err = ContractValidationError {
        tensor_name: "token_embedding".to_string(),
        rule_id: "F-DATA-QUALITY-001".to_string(),
        message: "94.5% zeros detected".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("F-DATA-QUALITY-001"));
    assert!(display.contains("token_embedding"));
    assert!(display.contains("94.5% zeros"));
}

#[test]
fn test_contract_validation_error_debug() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        message: "shape mismatch".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("ContractValidationError"));
}

#[test]
fn test_contract_validation_error_clone() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "F-DATA-QUALITY-002".to_string(),
        message: "NaN found".to_string(),
    };
    let cloned = err.clone();
    assert_eq!(cloned.tensor_name, "test");
    assert_eq!(cloned.rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_contract_validation_error_is_std_error() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "TEST".to_string(),
        message: "test error".to_string(),
    };
    // Verify it implements std::error::Error (via trait bound)
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_contract_validation_error_from_into_realizar_error() {
    let err = ContractValidationError {
        tensor_name: "embed".to_string(),
        rule_id: "F-DATA-QUALITY-001".to_string(),
        message: "too many zeros".to_string(),
    };
    let realizar_err: RealizarError = err.into();
    let msg = format!("{}", realizar_err);
    assert!(msg.contains("F-DATA-QUALITY-001"));
    assert!(msg.contains("embed"));
}

// ============================================================================
// validate_embedding edge cases
// ============================================================================

#[test]
fn test_validate_embedding_with_inf() {
    let vocab = 10;
    let dim = 8;
    let mut data: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
    data[5] = f32::INFINITY;
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Inf")));
}

#[test]
fn test_validate_embedding_zero_l2() {
    let vocab = 10;
    let dim = 8;
    let data = vec![1e-11f32; vocab * dim]; // Very small L2
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("L2")));
}

#[test]
fn test_validate_embedding_constant_values() {
    let vocab = 10;
    let dim = 8;
    let data = vec![0.5f32; vocab * dim]; // All identical
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("identical")));
}

// ============================================================================
// validate_weight edge cases
// ============================================================================

#[test]
fn test_validate_weight_shape_mismatch() {
    let data = vec![0.1f32; 100];
    let result = validate_weight("test", &data, 5, 5); // expects 25
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Shape")));
}

#[test]
fn test_validate_weight_nan() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::NAN;
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("NaN")));
}

#[test]
fn test_validate_weight_inf() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::INFINITY;
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Inf")));
}

#[test]
fn test_validate_weight_zero_l2() {
    let data = vec![1e-11f32; 100];
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("L2")));
}

#[test]
fn test_validate_weight_high_density_zeros() {
    let mut data = vec![0.0f32; 100];
    // Only 10% non-zero -> 90% zeros, exceeds 80% threshold
    for i in 90..100 {
        data[i] = 0.1;
    }
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("DENSITY")));
}

#[test]
fn test_validate_weight_good_data() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let result = validate_weight("test", &data, 10, 10);
    assert!(
        result.passed,
        "Good data should pass: {:?}",
        result.failures
    );
}

// ============================================================================
// validate_vector edge cases
// ============================================================================

#[test]
fn test_validate_vector_length_mismatch() {
    let data = vec![1.0f32; 50];
    let result = validate_vector("test", &data, 100);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Length")));
}

#[test]
fn test_validate_vector_nan() {
    let mut data = vec![1.0f32; 64];
    data[32] = f32::NAN;
    let result = validate_vector("test", &data, 64);
    assert!(!result.passed);
}

#[test]
fn test_validate_vector_inf() {
    let mut data = vec![1.0f32; 64];
    data[32] = f32::INFINITY;
    let result = validate_vector("test", &data, 64);
    assert!(!result.passed);
}

#[test]
fn test_validate_vector_good_data() {
    let data = vec![1.0f32; 64];
    let result = validate_vector("test", &data, 64);
    assert!(result.passed);
}

// ============================================================================
// enforce_embedding_validation and enforce_weight_validation
// ============================================================================

#[test]
fn test_enforce_embedding_validation_passes() {
    let vocab = 10;
    let dim = 8;
    let data: Vec<f32> = (0..vocab * dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let result = enforce_embedding_validation("test_emb", &data, vocab, dim);
    assert!(result.is_ok());
}

#[test]
fn test_enforce_embedding_validation_fails() {
    let data = vec![0.0f32; 100]; // All zeros
    let result = enforce_embedding_validation("bad_emb", &data, 10, 10);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("bad_emb"));
}

#[test]
fn test_enforce_weight_validation_passes() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let result = enforce_weight_validation("test_w", &data, 10, 10);
    assert!(result.is_ok());
}

#[test]
fn test_enforce_weight_validation_fails() {
    let data = vec![0.0f32; 100]; // All zeros -> density failure
    let result = enforce_weight_validation("bad_w", &data, 10, 10);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("bad_w"));
}

// ============================================================================
// ValidationResult
// ============================================================================

#[test]
fn test_validation_result_debug() {
    let result = ValidationResult {
        passed: true,
        stats: TensorStats::compute(&[1.0, 2.0]),
        failures: vec![],
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("ValidationResult"));
}

// ============================================================================
// ValidatedAprTransformer with optional biases
// ============================================================================

#[test]
fn test_validated_transformer_with_output_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.output_norm_bias = Some(vec![1.0; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid output_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_lm_head_bias() {
    let mut t = make_valid_transformer(1);
    t.lm_head_bias = Some(vec![0.1; 32]); // vocab_size = 32
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid lm_head_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_attn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].attn_norm_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid attn_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_qkv_bias() {
    let mut t = make_valid_transformer(1);
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    t.layers[0].qkv_bias = Some(vec![0.1; qkv_out_dim]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid qkv_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_attn_output_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].attn_output_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid attn_output_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_gate_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_gate_bias = Some(vec![0.1; 64]); // intermediate_dim = 64
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_gate_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_up_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_up_bias = Some(vec![0.1; 64]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_up_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_down_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_down_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_down_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_norm_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_rejects_nan_output_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.output_norm_bias = Some(vec![f32::NAN; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.tensor_name.contains("output_norm_bias"));
}

#[test]
fn test_validated_transformer_rejects_nan_lm_head_bias() {
    let mut t = make_valid_transformer(1);
    t.lm_head_bias = Some(vec![f32::NAN; 32]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.tensor_name.contains("lm_head_bias"));
}

#[test]
fn test_validated_transformer_rejects_nan_ffn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_norm_bias = Some(vec![f32::NAN; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_with_all_biases() {
    let mut t = make_valid_transformer(2);
    let hidden_dim = 16;
    let intermediate_dim = 64;
    let vocab_size = 32;
    let qkv_out_dim = hidden_dim + 2 * hidden_dim; // non-GQA

    t.output_norm_bias = Some(vec![0.1; hidden_dim]);
    t.lm_head_bias = Some(vec![0.1; vocab_size]);

    for layer in &mut t.layers {
        layer.attn_norm_bias = Some(vec![0.1; hidden_dim]);
        layer.qkv_bias = Some(vec![0.1; qkv_out_dim]);
        layer.attn_output_bias = Some(vec![0.1; hidden_dim]);
        layer.ffn_gate_bias = Some(vec![0.1; intermediate_dim]);
        layer.ffn_up_bias = Some(vec![0.1; intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.1; hidden_dim]);
        layer.ffn_norm_bias = Some(vec![0.1; hidden_dim]);
    }

    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "All-biases transformer should validate: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_rejects_wrong_output_norm() {
    let mut t = make_valid_transformer(1);
    t.output_norm_weight = vec![f32::NAN; 16];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_attn_output_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].attn_output_weight.len();
    t.layers[0].attn_output_weight = vec![0.0; len]; // all zeros
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_ffn_gate_weight() {
    let mut t = make_valid_transformer(1);
    if let Some(ref mut w) = t.layers[0].ffn_gate_weight {
        for v in w.iter_mut() {
            *v = 0.0;
        }
    }
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_ffn_up_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].ffn_up_weight.len();
    t.layers[0].ffn_up_weight = vec![0.0; len];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_ffn_down_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].ffn_down_weight.len();
    t.layers[0].ffn_down_weight = vec![0.0; len];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_config_accessor() {
    let t = make_valid_transformer(2);
    let validated = ValidatedAprTransformer::validate(t).expect("should pass");
    assert_eq!(validated.config().hidden_dim, 16);
    assert_eq!(validated.config().num_layers, 2);
}

#[test]
fn test_validated_transformer_with_zero_num_heads() {
    // Edge case: num_heads = 0 should use hidden_dim as head_dim fallback
    let mut t = make_valid_transformer(1);
    t.config.num_heads = 0;
    t.config.num_kv_heads = 0;
    // This will fail on shape mismatch since QKV dimensions change
    let result = ValidatedAprTransformer::validate(t);
    // Should error due to shape mismatch, not panic
    assert!(result.is_err());
}

// ============================================================================
// Helpers
// ============================================================================

fn make_valid_transformer(num_layers: usize) -> AprTransformer {
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 4;
    let vocab_size = 32;
    let intermediate_dim = 64;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let make_data = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect()
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
        },
        token_embedding: make_data(vocab_size * hidden_dim),
        layers: (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: make_data(qkv_out_dim * hidden_dim),
                qkv_bias: None,
                attn_output_weight: make_data(hidden_dim * hidden_dim),
                attn_output_bias: None,
                ffn_gate_weight: Some(make_data(intermediate_dim * hidden_dim)),
                ffn_gate_bias: None,
                ffn_up_weight: make_data(intermediate_dim * hidden_dim),
                ffn_up_bias: None,
                ffn_down_weight: make_data(hidden_dim * intermediate_dim),
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden_dim]),
                ffn_norm_bias: None,
            })
            .collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_data(vocab_size * hidden_dim),
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}
