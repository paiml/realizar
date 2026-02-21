
// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LEGACY FUNCTION TESTS (validate_embedding, etc.)
    // =========================================================================

    #[test]
    fn test_validates_good_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(
            result.passed,
            "Good embedding should pass: {:?}",
            result.failures
        );
    }

    #[test]
    fn test_rejects_mostly_zero_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (vocab_size * 95 / 100 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "95% zero embedding should fail");
        assert!(result.failures.iter().any(|f| f.contains("DENSITY")));
    }

    #[test]
    fn test_rejects_nan_embedding() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = vec![0.1; vocab_size * hidden_dim];
        data[5] = f32::NAN;

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "NaN embedding should fail");
    }

    #[test]
    fn test_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = validate_embedding("test", &data, 100, 64);
        assert!(!result.passed, "Wrong shape should fail");
    }

    // =========================================================================
    // POPPERIAN FALSIFICATION TESTS FOR NEWTYPES (PMAT-235)
    // Per Popper (1959), these attempt to DISPROVE the contract works.
    // =========================================================================

    #[test]
    fn falsify_001_validated_embedding_rejects_all_zeros() {
        let bad_data = vec![0.0f32; 100 * 64];
        let result = ValidatedEmbedding::new(bad_data, 100, 64);
        assert!(result.is_err(), "Should reject 100% zeros");
        let err = result.unwrap_err();
        assert!(err.message.contains("DENSITY"), "Error: {}", err.message);
    }

    #[test]
    fn falsify_001_validated_embedding_rejects_94pct_zeros() {
        // Simulate PMAT-234 bug
        let vocab_size = 1000;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (945 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject 94.5% zeros");
    }

    #[test]
    fn falsify_001_validated_embedding_accepts_good_data() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(
            result.is_ok(),
            "Should accept good data: {:?}",
            result.err()
        );
    }

    #[test]
    fn falsify_003_validated_embedding_rejects_nan() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        data[5] = f32::NAN;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject NaN");
    }

    #[test]
    fn falsify_004_spot_check_catches_offset_bug() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10% (token 10)
        let token_10_start = 10 * hidden_dim;
        for i in token_10_start..(token_10_start + hidden_dim) {
            data[i] = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should catch zero token at 10%");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-004");
    }

    #[test]
    fn falsify_005_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = ValidatedEmbedding::new(data, 100, 64);
        assert!(result.is_err(), "Should reject wrong shape");
    }

    #[test]
    fn validated_weight_rejects_all_zeros() {
        let data = vec![0.0f32; 100];
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_weight_accepts_good_data() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn validated_vector_rejects_wrong_length() {
        let data = vec![0.1f32; 50];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_vector_accepts_good_data() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_ok());
    }

    // =========================================================================
    // VALIDATED APR TRANSFORMER FALSIFICATION TESTS (PMAT-235)
    // =========================================================================

    /// Helper: create a valid AprTransformer for testing
    fn make_valid_transformer(num_layers: usize) -> AprTransformer {
        use crate::apr_transformer::{AprTransformerConfig, AprTransformerLayer};

        let hidden_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 4;
        let vocab_size = 32;
        let intermediate_dim = 64;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        let config = AprTransformerConfig {
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
        };

        // Non-zero sin pattern data
        let make_data = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
                .collect()
        };

        let layers = (0..num_layers)
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
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: make_data(vocab_size * hidden_dim),
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: make_data(vocab_size * hidden_dim),
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_embedding() {
        let mut t = make_valid_transformer(1);
        t.token_embedding[5] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in embedding");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("embedding"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_zero_layer_weight() {
        let mut t = make_valid_transformer(1);
        // Zero out entire qkv_weight → density gate
        let len = t.layers[0].qkv_weight.len();
        t.layers[0].qkv_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject all-zero qkv_weight");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("qkv_weight"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_in_deep_layer() {
        let mut t = make_valid_transformer(4);
        t.layers[3].ffn_up_weight[0] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in layer 3 ffn_up");
        let err = result.unwrap_err();
        assert!(
            err.tensor_name.contains("layers.3.ffn_up_weight"),
            "Error: {err}"
        );
    }

    #[test]
    fn falsify_validated_transformer_identifies_tensor_name() {
        let mut t = make_valid_transformer(2);
        // Corrupt lm_head_weight
        let len = t.lm_head_weight.len();
        t.lm_head_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.tensor_name, "lm_head_weight",
            "Error should name the tensor: {err}"
        );
    }

    #[test]
    fn validated_transformer_accepts_good_model() {
        let t = make_valid_transformer(2);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_ok(), "Good model should pass: {:?}", result.err());
    }

    #[test]
    fn validated_transformer_deref_transparent_access() {
        let t = make_valid_transformer(1);
        let validated = ValidatedAprTransformer::validate(t).expect("validation should pass");

        // Access config through Deref
        assert_eq!(validated.config.hidden_dim, 16);
        assert_eq!(validated.config.num_layers, 1);
        assert_eq!(validated.config.vocab_size, 32);

        // Access fields through Deref
        assert!(!validated.token_embedding.is_empty());
        assert_eq!(validated.layers.len(), 1);

        // Access through explicit methods
        assert_eq!(validated.config().hidden_dim, 16);
        assert_eq!(validated.transformer().config.num_layers, 1);

        // into_inner works
        let inner = validated.into_inner();
        assert_eq!(inner.config.hidden_dim, 16);
    }

    // =========================================================================
    // GH-46 FALSIFICATION: Rosetta strict validation boundaries
    // =========================================================================

    /// GH-46: Embedding density gate must reject >50% zeros.
    /// Before the fix, validation was too lenient and passed all-zero embeddings.
    #[test]
    fn test_falsify_gh46_embedding_density_threshold_50pct() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let total = vocab * hidden;
        // 51% zeros — must FAIL
        let zero_count = (total as f64 * 0.51).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_embedding("test_embed", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: >50% zero embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: Weight density gate must reject >80% zeros.
    /// Before the fix, weights with mostly zeros passed validation silently.
    #[test]
    fn test_falsify_gh46_weight_density_threshold_80pct() {
        let rows = 32_usize;
        let cols = 16_usize;
        let total = rows * cols;
        // 81% zeros — must FAIL
        let zero_count = (total as f64 * 0.81).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_weight("test_weight", &data, rows, cols);
        assert!(
            !result.passed,
            "GH-46: >80% zero weight must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: L2 norm gate must reject flat (constant) tensors.
    /// A constant tensor has zero variance — signals import corruption.
    #[test]
    fn test_falsify_gh46_rejects_flat_tensor() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        // All identical non-zero values — L2 norm > 0 but max-min == 0
        let data = vec![0.5_f32; vocab * hidden];
        let result = validate_embedding("test_flat", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Flat (constant) embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: NaN gate must catch even a single NaN in embeddings.
    /// Before strict validation, NaN could propagate through inference.
    #[test]
    fn test_falsify_gh46_single_nan_detected() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let mut data: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        // Inject single NaN at arbitrary position
        data[vocab * hidden / 2] = f32::NAN;
        let result = validate_embedding("test_nan", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Single NaN must be caught by validation"
        );
        assert!(
            result
                .failures
                .iter()
                .any(|f| f.to_lowercase().contains("nan")),
            "GH-46: Failure message must mention NaN"
        );
    }

    // =========================================================================
    // PMAT-299: Architecture completeness falsification tests
    // =========================================================================

    /// FALSIFY-GAP4-001: Qwen3 without QK norm MUST be rejected.
    /// This is the exact GH-279 root cause — Qwen3 missing attn_q_norm produced garbage.
    #[test]
    fn falsify_gap4_001_qwen3_without_qk_norm_rejected() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "qwen3".to_string();
        // QK norm is None — Qwen3 REQUIRES it
        assert!(t.layers[0].attn_q_norm_weight.is_none());
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Qwen3 without QK norm must be rejected");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("QK norm") || err.message.contains("attn_q_norm"),
            "Error must mention QK norm: {}",
            err.message
        );
    }

    /// FALSIFY-GAP4-002: Qwen3 WITH QK norm passes.
    #[test]
    fn falsify_gap4_002_qwen3_with_qk_norm_passes() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "qwen3".to_string();
        let head_dim = t.config.hidden_dim / t.config.num_heads;
        for layer in &mut t.layers {
            layer.attn_q_norm_weight = Some(vec![1.0; head_dim]);
            layer.attn_k_norm_weight = Some(vec![1.0; head_dim]);
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(
            result.is_ok(),
            "Qwen3 with QK norm should pass: {:?}",
            result.err()
        );
    }

    /// FALSIFY-GAP4-003: Qwen2 without bias MUST be rejected.
    #[test]
    fn falsify_gap4_003_qwen2_without_bias_rejected() {
        let mut t = make_valid_transformer(1);
        t.config.architecture = "qwen2".to_string();
        assert!(t.layers[0].qkv_bias.is_none());
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Qwen2 without bias must be rejected");
    }

    /// FALSIFY-GAP4-004: LLaMA without QK norm or bias is fine.
    #[test]
    fn falsify_gap4_004_llama_without_optional_passes() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "llama".to_string();
        let result = ValidatedAprTransformer::validate(t);
        assert!(
            result.is_ok(),
            "LLaMA should pass without QK norm or bias: {:?}",
            result.err()
        );
    }

    /// FALSIFY-GAP4-005: Missing QK norm detected on ANY layer, not just layer 0.
    #[test]
    fn falsify_gap4_005_qwen3_missing_norm_on_later_layer() {
        let mut t = make_valid_transformer(4);
        t.config.architecture = "qwen3".to_string();
        let head_dim = t.config.hidden_dim / t.config.num_heads;
        for layer in &mut t.layers {
            layer.attn_q_norm_weight = Some(vec![1.0; head_dim]);
            layer.attn_k_norm_weight = Some(vec![1.0; head_dim]);
        }
        // Remove QK norm from layer 3 only
        t.layers[3].attn_k_norm_weight = None;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Must catch missing QK norm on layer 3");
        let err = result.unwrap_err();
        assert!(
            err.tensor_name.contains("3"),
            "Error must identify layer 3: {}",
            err.tensor_name
        );
    }
}

// T-COV-95 Coverage Bridge (Part 02 - Accessors, error paths, optional biases)
#[cfg(test)]
#[path = "validation_tests_02.rs"]
mod validation_tests_02;
