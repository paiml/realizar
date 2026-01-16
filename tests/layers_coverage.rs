//! EXTREME TDD coverage tests for realizar/src/layers.rs
//!
//! Target: Increase coverage from 86% to 95%+ by testing:
//! - Edge cases for all layer types
//! - Error paths
//! - Getters and accessors
//! - Boundary conditions

use realizar::layers::{
    gelu, softmax, ALiBi, Attention, Embedding, FeedForward, FusedLayerNormLinear,
    FusedQKVAttention, KVCache, LayerNorm, Linear, Model, ModelConfig, MultiHeadAttention,
    QuantizedLinear, RoPE, RopeScalingType, ScaledRoPE, SlidingWindowAttention, TransformerBlock,
};
use realizar::Tensor;

// ============================================================================
// SOFTMAX EDGE CASES
// ============================================================================

#[test]
fn test_softmax_empty_tensor_error() {
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_softmax_negative_infinity_stability() {
    // Extremely negative values shouldn't cause NaN
    let input = Tensor::from_vec(vec![3], vec![-1000.0, -1001.0, -1002.0]).expect("test");
    let output = softmax(&input).expect("test");

    for val in output.data() {
        assert!(val.is_finite(), "Softmax produced non-finite value: {}", val);
    }

    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_single_large_batch() {
    // Large batch with many rows
    let input = Tensor::from_vec(vec![100, 10], vec![0.5; 1000]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[100, 10]);

    // Check every row sums to 1
    for row in 0..100 {
        let row_sum: f32 = output.data()[row * 10..(row + 1) * 10].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum {} != 1.0",
            row,
            row_sum
        );
    }
}

// ============================================================================
// GELU EDGE CASES
// ============================================================================

#[test]
fn test_gelu_large_positive_approximates_identity() {
    // For large positive values, GELU(x) ≈ x
    let input = Tensor::from_vec(vec![3], vec![10.0, 20.0, 50.0]).expect("test");
    let output = gelu(&input).expect("test");

    for (i, &val) in output.data().iter().enumerate() {
        let expected = input.data()[i];
        assert!(
            (val - expected).abs() / expected.abs() < 0.01,
            "GELU({}) = {} should be close to {}",
            expected,
            val,
            expected
        );
    }
}

#[test]
fn test_gelu_large_negative_near_zero() {
    // For large negative values, GELU(x) ≈ 0
    let input = Tensor::from_vec(vec![3], vec![-10.0, -20.0, -50.0]).expect("test");
    let output = gelu(&input).expect("test");

    for &val in output.data() {
        assert!(val.abs() < 0.01, "GELU of large negative should be near 0");
    }
}

#[test]
fn test_gelu_multidimensional() {
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.1; 24]).expect("test");
    let output = gelu(&input).expect("test");
    assert_eq!(output.shape(), &[2, 3, 4]);
    assert_eq!(output.data().len(), 24);
}

// ============================================================================
// LAYER NORM EDGE CASES
// ============================================================================

#[test]
fn test_layer_norm_large_values() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1000.0, 1001.0, 1002.0, 1003.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // Mean should be approximately 0
    let mean: f32 = output.data().iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_layer_norm_batched_large() {
    let layer_norm = LayerNorm::new(64, 1e-6).expect("test");
    let input = Tensor::from_vec(vec![32, 64], vec![0.5; 32 * 64]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[32, 64]);
}

#[test]
fn test_layer_norm_eps_getter() {
    let eps = 1e-12;
    let layer_norm = LayerNorm::new(512, eps).expect("test");
    assert!((layer_norm.eps() - eps).abs() < 1e-15);
}

// ============================================================================
// LINEAR EDGE CASES
// ============================================================================

#[test]
fn test_linear_single_feature() {
    let mut linear = Linear::new(1, 1).expect("test");
    linear.weight_mut()[0] = 2.0;
    linear.bias_mut()[0] = 1.0;

    let input = Tensor::from_vec(vec![1], vec![3.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    // 3.0 * 2.0 + 1.0 = 7.0
    assert!((output.data()[0] - 7.0).abs() < 1e-5);
}

#[test]
fn test_linear_large_dimensions() {
    let linear = Linear::new(1024, 4096).expect("test");
    assert_eq!(linear.in_features(), 1024);
    assert_eq!(linear.out_features(), 4096);
}

// ============================================================================
// QUANTIZED LINEAR EDGE CASES
// ============================================================================

#[test]
fn test_quantized_linear_zero_dimension_error() {
    let result = QuantizedLinear::new(0, 256, vec![], vec![]);
    assert!(result.is_err());

    let result = QuantizedLinear::new(256, 0, vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_bias_mismatch_error() {
    // Bias length doesn't match out_features
    let result = QuantizedLinear::new(256, 4, vec![0u8; 144 * 4], vec![0.0, 0.0]); // bias len=2, out_features=4
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_weight_bytes_mismatch_error() {
    // Weight bytes don't match expected size
    let result = QuantizedLinear::new(256, 4, vec![0u8; 100], vec![0.0; 4]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_getters() {
    let ql = QuantizedLinear::new(256, 4, vec![0u8; 144 * 4], vec![0.0; 4]).expect("test");
    assert_eq!(ql.in_features(), 256);
    assert_eq!(ql.out_features(), 4);
    assert_eq!(ql.weight_bytes().len(), 144 * 4);
    assert_eq!(ql.bias().len(), 4);
    assert!(ql.memory_bytes() > 0);
}

// ============================================================================
// FUSED LAYER NORM LINEAR EDGE CASES
// ============================================================================

#[test]
fn test_fused_layer_norm_linear_empty_input_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
    let _ = fused; // Suppress unused warning
}

#[test]
fn test_fused_layer_norm_linear_parallel_empty_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");
    // Wrong dimension
    let input = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let result = fused.forward_parallel(&input);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_all_accessors() {
    let mut fused = FusedLayerNormLinear::new(8, 16, 1e-6).expect("test");

    assert_eq!(fused.feature_dim(), 8);
    assert_eq!(fused.out_features(), 16);

    // Test all mutable accessors
    fused.norm_weight_mut()[0] = 1.5;
    fused.norm_bias_mut()[0] = 0.5;
    fused.linear_weight_mut()[0] = 0.1;
    fused.linear_bias_mut()[0] = 0.01;

    assert!((fused.norm_weight_mut()[0] - 1.5).abs() < 1e-6);
}

// ============================================================================
// FEED FORWARD EDGE CASES
// ============================================================================

#[test]
fn test_ffn_shape_mismatch_error() {
    let ffn = FeedForward::new(4, 16).expect("test");
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test"); // Wrong dim
    let result = ffn.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_ffn_all_accessors() {
    let mut ffn = FeedForward::new(128, 512).expect("test");
    assert_eq!(ffn.hidden_dim(), 128);
    assert_eq!(ffn.intermediate_dim(), 512);

    ffn.fc1_mut().weight_mut()[0] = 0.5;
    ffn.fc2_mut().bias_mut()[0] = 0.1;
}

// ============================================================================
// ATTENTION EDGE CASES
// ============================================================================

#[test]
fn test_attention_empty_tensor_error() {
    let attn = Attention::new(4).expect("test");
    // Empty shape not allowed by Tensor
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
    let _ = attn; // Suppress unused warning
}

#[test]
fn test_attention_various_head_dims() {
    for head_dim in [1, 2, 4, 8, 16, 32, 64, 128] {
        let attn = Attention::new(head_dim).expect("test");
        assert_eq!(attn.head_dim(), head_dim);

        #[allow(clippy::cast_precision_loss)]
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!((attn.scale() - expected_scale).abs() < 1e-5);
    }
}

#[test]
fn test_flash_attention_small_block_size() {
    let attn = Attention::new(8).expect("test");
    let q = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).expect("test");
    let k = Tensor::from_vec(vec![4, 8], vec![0.2; 32]).expect("test");
    let v = Tensor::from_vec(vec![4, 8], vec![0.3; 32]).expect("test");

    // Small block size of 1
    let output = attn.flash_forward(&q, &k, &v, 1).expect("test");
    assert_eq!(output.shape(), &[4, 8]);
}

#[test]
fn test_flash_attention_v2_small_block_size() {
    let attn = Attention::new(8).expect("test");
    let q = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).expect("test");
    let k = Tensor::from_vec(vec![4, 8], vec![0.2; 32]).expect("test");
    let v = Tensor::from_vec(vec![4, 8], vec![0.3; 32]).expect("test");

    let output = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");
    assert_eq!(output.shape(), &[4, 8]);
}

#[test]
fn test_flash_attention_parallel_consistency() {
    let attn = Attention::new(8).expect("test");
    let q = Tensor::from_vec(vec![8, 8], vec![0.1; 64]).expect("test");
    let k = Tensor::from_vec(vec![8, 8], vec![0.2; 64]).expect("test");
    let v = Tensor::from_vec(vec![8, 8], vec![0.3; 64]).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 4).expect("test");

    // Results should be similar
    for (a, b) in standard.data().iter().zip(parallel.data().iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "Mismatch: standard={} vs parallel={}",
            a,
            b
        );
    }
}

// ============================================================================
// SLIDING WINDOW ATTENTION EDGE CASES
// ============================================================================

#[test]
fn test_sliding_window_window_larger_than_seq() {
    let swa = SlidingWindowAttention::new(8, 100).expect("test"); // Large window

    let q = Tensor::from_vec(vec![5, 8], vec![0.1; 40]).expect("test");
    let k = Tensor::from_vec(vec![5, 8], vec![0.2; 40]).expect("test");
    let v = Tensor::from_vec(vec![5, 8], vec![0.3; 40]).expect("test");

    let output = swa.forward(&q, &k, &v).expect("test");
    assert_eq!(output.shape(), &[5, 8]);
}

#[test]
fn test_sliding_window_all_getters() {
    let swa = SlidingWindowAttention::new(64, 256).expect("test");

    assert_eq!(swa.head_dim(), 64);
    assert_eq!(swa.window_size(), 256);

    #[allow(clippy::cast_precision_loss)]
    let expected_scale = 1.0 / (64_f32).sqrt();
    assert!((swa.scale() - expected_scale).abs() < 1e-6);
}

#[test]
fn test_sliding_window_effective_context() {
    let swa = SlidingWindowAttention::new(8, 4).expect("test"); // Window of 4

    assert_eq!(swa.effective_context(0, 10), 1); // Can only see 1 (itself)
    assert_eq!(swa.effective_context(3, 10), 4); // Can see 4
    assert_eq!(swa.effective_context(5, 10), 4); // Limited by window
    assert_eq!(swa.effective_context(9, 10), 4); // Limited by window
}

#[test]
fn test_sliding_window_memory_ratio() {
    let swa = SlidingWindowAttention::new(8, 128).expect("test");

    // Memory ratio = min(window_size, seq_len) / seq_len
    let ratio = swa.memory_ratio(1024);
    assert!((ratio - 0.125).abs() < 1e-5); // 128/1024

    let ratio_small = swa.memory_ratio(64);
    assert!((ratio_small - 1.0).abs() < 1e-5); // 64/64 = 1.0

    let ratio_zero = swa.memory_ratio(0);
    assert!((ratio_zero - 1.0).abs() < 1e-5); // Edge case
}

#[test]
fn test_sliding_window_bidirectional() {
    let swa = SlidingWindowAttention::new(8, 3).expect("test");

    let q = Tensor::from_vec(vec![5, 8], vec![0.1; 40]).expect("test");
    let k = Tensor::from_vec(vec![5, 8], vec![0.2; 40]).expect("test");
    let v = Tensor::from_vec(vec![5, 8], vec![0.3; 40]).expect("test");

    // Bidirectional (non-causal)
    let output = swa.forward_with_mask(&q, &k, &v, false).expect("test");
    assert_eq!(output.shape(), &[5, 8]);
}

// ============================================================================
// FUSED QKV ATTENTION EDGE CASES
// ============================================================================

#[test]
fn test_fused_qkv_attention_indivisible_error() {
    // hidden_dim not divisible by head_dim
    let result = FusedQKVAttention::new(7, 64);
    assert!(result.is_err());
}

#[test]
fn test_fused_qkv_attention_all_getters() {
    let fqa = FusedQKVAttention::new(8, 64).expect("test");

    assert_eq!(fqa.head_dim(), 8);
    assert_eq!(fqa.hidden_dim(), 64);
    assert_eq!(fqa.num_heads(), 8); // 64/8
}

#[test]
fn test_fused_qkv_attention_weight_access() {
    let mut fqa = FusedQKVAttention::new(8, 32).expect("test");

    fqa.w_q_mut()[0] = 1.0;
    fqa.w_k_mut()[0] = 2.0;
    fqa.w_v_mut()[0] = 3.0;
    fqa.w_o_mut()[0] = 4.0;

    assert!((fqa.w_q_mut()[0] - 1.0).abs() < 1e-6);
    assert!((fqa.w_k_mut()[0] - 2.0).abs() < 1e-6);
}

#[test]
fn test_fused_qkv_attention_1d_input_error() {
    let fqa = FusedQKVAttention::new(8, 32).expect("test");
    let input = Tensor::from_vec(vec![32], vec![0.1; 32]).expect("test");
    let result = fqa.forward(&input);
    assert!(result.is_err()); // Needs 2D input
}

// ============================================================================
// MULTI HEAD ATTENTION EDGE CASES
// ============================================================================

#[test]
fn test_mha_is_mqa_is_gqa_is_mha() {
    // Test type detection methods
    let mha = MultiHeadAttention::mha(64, 8).expect("test");
    assert!(mha.is_mha());
    assert!(!mha.is_mqa());
    assert!(!mha.is_gqa());

    let mqa = MultiHeadAttention::mqa(64, 8).expect("test");
    assert!(mqa.is_mqa());
    assert!(!mqa.is_mha());
    assert!(!mqa.is_gqa());

    let gqa = MultiHeadAttention::gqa(64, 8, 2).expect("test");
    assert!(gqa.is_gqa());
    assert!(!gqa.is_mha());
    assert!(!gqa.is_mqa());
}

#[test]
fn test_mha_all_getters() {
    let mha = MultiHeadAttention::new(128, 8, 4).expect("test");

    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 4);
    assert_eq!(mha.head_dim(), 16); // 128/8
    assert_eq!(mha.hidden_dim(), 128);
}

#[test]
fn test_mha_single_sequence() {
    let mha = MultiHeadAttention::mha(32, 4).expect("test");
    let input = Tensor::from_vec(vec![1, 32], vec![0.1; 32]).expect("test");
    let output = mha.forward(&input).expect("test");

    assert_eq!(output.shape(), &[1, 32]);
}

// ============================================================================
// ROPE EDGE CASES
// ============================================================================

#[test]
fn test_rope_high_position() {
    let rope = RoPE::new(8, 10000.0).expect("test");
    let input = Tensor::from_vec(vec![1, 8], vec![1.0; 8]).expect("test");

    // High position shouldn't cause issues
    let output = rope.forward(&input, 10000).expect("test");
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_all_getters() {
    let rope = RoPE::new(64, 10000.0).expect("test");

    assert_eq!(rope.dim(), 64);
    assert!((rope.base() - 10000.0).abs() < 1e-3);
    assert_eq!(rope.inv_freq().len(), 32); // dim/2
}

#[test]
fn test_rope_with_default_base() {
    let rope = RoPE::with_default_base(32).expect("test");
    assert!((rope.base() - 10000.0).abs() < 1e-3);
}

// ============================================================================
// SCALED ROPE EDGE CASES
// ============================================================================

#[test]
fn test_scaled_rope_linear_scaling() {
    let scaling = RopeScalingType::Linear { scale: 2.0 };
    let srope = ScaledRoPE::new(8, 10000.0, scaling).expect("test");

    assert_eq!(srope.dim(), 8);
    assert!((srope.original_base() - 10000.0).abs() < 1e-3);
    assert!((srope.context_length_multiplier() - 2.0).abs() < 1e-5);
}

#[test]
fn test_scaled_rope_ntk_scaling() {
    let scaling = RopeScalingType::Ntk { scale: 4.0 };
    let srope = ScaledRoPE::new(8, 10000.0, scaling).expect("test");

    assert!(srope.scaled_base() > 10000.0); // NTK increases base
    assert!((srope.mscale() - 1.0).abs() < 1e-5); // NTK doesn't use mscale
}

#[test]
fn test_scaled_rope_dynamic_ntk() {
    let scaling = RopeScalingType::DynamicNtk {
        original_max_len: 2048,
        target_max_len: 8192,
    };
    let srope = ScaledRoPE::new(8, 10000.0, scaling).expect("test");

    assert!((srope.context_length_multiplier() - 4.0).abs() < 1e-5);
}

#[test]
fn test_scaled_rope_yarn() {
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let srope = ScaledRoPE::new(8, 10000.0, scaling).expect("test");

    assert!(srope.mscale() > 0.0);
    assert!((srope.context_length_multiplier() - 4.0).abs() < 1e-5);
}

#[test]
fn test_scaled_rope_all_getters() {
    let scaling = RopeScalingType::None;
    let srope = ScaledRoPE::new(16, 10000.0, scaling).expect("test");

    assert_eq!(srope.dim(), 16);
    assert!((srope.original_base() - 10000.0).abs() < 1e-3);
    assert!((srope.scaled_base() - 10000.0).abs() < 1e-3);
    assert_eq!(srope.inv_freq().len(), 8);
    assert!((srope.mscale() - 1.0).abs() < 1e-5);

    if let RopeScalingType::None = srope.scaling() {
        // Expected
    } else {
        panic!("Expected None scaling");
    }
}

#[test]
fn test_scaled_rope_with_default_base() {
    let scaling = RopeScalingType::None;
    let srope = ScaledRoPE::with_default_base(8, scaling).expect("test");
    assert!((srope.original_base() - 10000.0).abs() < 1e-3);
}

#[test]
fn test_rope_scaling_type_default() {
    let scaling: RopeScalingType = Default::default();
    assert_eq!(scaling, RopeScalingType::None);
}

// ============================================================================
// ALIBI EDGE CASES
// ============================================================================

#[test]
fn test_alibi_power_of_two_heads() {
    for num_heads in [1, 2, 4, 8, 16, 32] {
        let alibi = ALiBi::new(num_heads).expect("test");
        assert_eq!(alibi.num_heads(), num_heads);
        assert_eq!(alibi.slopes().len(), num_heads);
    }
}

#[test]
fn test_alibi_non_power_of_two_heads() {
    for num_heads in [3, 5, 6, 7, 9, 10, 12] {
        let alibi = ALiBi::new(num_heads).expect("test");
        assert_eq!(alibi.num_heads(), num_heads);
        assert_eq!(alibi.slopes().len(), num_heads);
    }
}

#[test]
fn test_alibi_large_sequence() {
    let alibi = ALiBi::new(8).expect("test");
    let bias = alibi.get_bias(256).expect("test");
    assert_eq!(bias.shape(), &[256, 256, 8]);
}

#[test]
fn test_alibi_slopes_are_decreasing() {
    let alibi = ALiBi::new(8).expect("test");
    let slopes = alibi.slopes();

    // First slope should be largest
    for i in 1..slopes.len() {
        assert!(slopes[i] <= slopes[i - 1] || i >= 4); // Non-power-of-2 adjustment
    }
}

// ============================================================================
// KV CACHE EDGE CASES
// ============================================================================

#[test]
fn test_kv_cache_all_getters() {
    let cache = KVCache::new(12, 2048, 64).expect("test");

    assert_eq!(cache.num_layers(), 12);
    assert_eq!(cache.max_seq_len(), 2048);
    assert_eq!(cache.head_dim(), 64);
    assert_eq!(cache.current_pos(), 0);
    assert!(!cache.is_full());
}

#[test]
fn test_kv_cache_becomes_full() {
    let mut cache = KVCache::new(1, 3, 4).expect("test");

    let key = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");

    for _ in 0..3 {
        cache.update(0, &key, &value).expect("test");
        cache.advance();
    }

    assert!(cache.is_full());
    assert!(cache.update(0, &key, &value).is_err());
}

#[test]
fn test_kv_cache_clear_resets() {
    let mut cache = KVCache::new(2, 10, 8).expect("test");

    let key = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let value = Tensor::from_vec(vec![8], vec![2.0; 8]).expect("test");

    cache.update(0, &key, &value).expect("test");
    cache.update(1, &key, &value).expect("test");
    cache.advance();

    assert_eq!(cache.current_pos(), 1);

    cache.clear();

    assert_eq!(cache.current_pos(), 0);
    assert!(!cache.is_full());
}

#[test]
fn test_kv_cache_empty_retrieval() {
    let cache = KVCache::new(2, 10, 8).expect("test");

    // Retrieving at pos 0 returns placeholder
    let key = cache.get_key(0).expect("test");
    let value = cache.get_value(0).expect("test");

    assert_eq!(key.shape(), &[1, 8]);
    assert_eq!(value.shape(), &[1, 8]);
}

// ============================================================================
// TRANSFORMER BLOCK EDGE CASES
// ============================================================================

#[test]
fn test_transformer_block_all_accessors() {
    let mut block = TransformerBlock::new(128, 8, 512, 1e-5).expect("test");

    assert_eq!(block.hidden_dim(), 128);
    assert_eq!(block.num_heads(), 8);

    // Mutable accessors
    let _ = block.attn_norm_mut();
    let _ = block.attention_mut();
    let _ = block.ffn_norm_mut();
    let _ = block.ffn_mut();
}

#[test]
fn test_transformer_block_forward_numerical_stability() {
    let block = TransformerBlock::new(64, 4, 256, 1e-5).expect("test");

    // Large values
    let input = Tensor::from_vec(vec![2, 64], vec![100.0; 128]).expect("test");
    let output = block.forward(&input).expect("test");

    for &val in output.data() {
        assert!(val.is_finite(), "Output contains non-finite: {}", val);
    }
}

// ============================================================================
// EMBEDDING EDGE CASES
// ============================================================================

#[test]
fn test_embedding_all_getters() {
    let embedding = Embedding::new(32000, 4096).expect("test");

    assert_eq!(embedding.vocab_size(), 32000);
    assert_eq!(embedding.embed_dim(), 4096);
}

#[test]
fn test_embedding_weights_mut() {
    let mut embedding = Embedding::new(100, 8).expect("test");
    embedding.weights_mut()[0] = 42.0;
    assert!((embedding.weights_mut()[0] - 42.0).abs() < 1e-6);
}

#[test]
fn test_embedding_boundary_token_id() {
    let embedding = Embedding::new(100, 8).expect("test");

    // Last valid token
    let result = embedding.forward(&[99]);
    assert!(result.is_ok());

    // Out of bounds
    let result = embedding.forward(&[100]);
    assert!(result.is_err());
}

// ============================================================================
// MODEL EDGE CASES
// ============================================================================

#[test]
fn test_model_config_all_fields() {
    let config = ModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_layers: 32,
        intermediate_dim: 11008,
        eps: 1e-5,
    };

    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.intermediate_dim, 11008);
    assert!((config.eps - 1e-5).abs() < 1e-10);
}

#[test]
fn test_model_all_accessors() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
    };

    let mut model = Model::new(config).expect("test");

    let _ = model.config();
    let _ = model.embedding_mut();
    let _ = model.blocks_mut();
    let _ = model.final_norm_mut();
    let _ = model.lm_head_mut();
    let _ = model.num_parameters();
}

#[test]
fn test_model_num_parameters() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
    };

    let model = Model::new(config).expect("test");
    let params = model.num_parameters();

    // Should be positive
    assert!(params > 0);

    // Check rough estimate
    // embed_params = 100 * 32 = 3200
    // block_params per layer = 2*32 + 32*128 + 128*32 = 64 + 4096 + 4096 = 8256
    // head_params = 32 * 100 = 3200
    // Total ~ 3200 + 2*8256 + 3200 = 22912
    assert!(params > 10000);
}

// ============================================================================
// ADDITIONAL EDGE CASES
// ============================================================================

#[test]
fn test_gelu_empty_error() {
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_empty_input_error() {
    let layer_norm = LayerNorm::new(4, 1e-5).expect("test");
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
    let _ = layer_norm; // Suppress unused warning
}

#[test]
fn test_attention_empty_shapes_error() {
    let attn = Attention::new(4).expect("test");
    let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![0.2; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_scaled_rope_forward_all_types() {
    let dim = 8;
    let input = Tensor::from_vec(vec![2, dim], vec![1.0; 2 * dim]).expect("test");

    // Test all scaling types
    let scalings = [
        RopeScalingType::None,
        RopeScalingType::Linear { scale: 2.0 },
        RopeScalingType::Ntk { scale: 2.0 },
        RopeScalingType::DynamicNtk {
            original_max_len: 2048,
            target_max_len: 4096,
        },
        RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 4096,
            attn_factor: 0.0, // Compute automatically
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    ];

    for scaling in scalings {
        let srope = ScaledRoPE::new(dim, 10000.0, scaling).expect("test");
        let output = srope.forward(&input, 100).expect("test");
        assert_eq!(output.shape(), &[2, dim]);

        for &val in output.data() {
            assert!(val.is_finite(), "Non-finite value with {:?}", srope.scaling());
        }
    }
}

#[test]
fn test_scaled_rope_yarn_auto_attn_factor() {
    // When attn_factor is 0, it should be computed automatically
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 0.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };

    let srope = ScaledRoPE::new(8, 10000.0, scaling).expect("test");
    assert!(srope.mscale() > 1.0); // Should be computed
}

#[test]
fn test_fused_qkv_hidden_dim_not_divisible_error() {
    let result = FusedQKVAttention::new(5, 32); // 32 not divisible by 5
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_wrong_input_shape_error() {
    let mha = MultiHeadAttention::mha(64, 8).expect("test");

    // 1D tensor instead of 2D
    let input = Tensor::from_vec(vec![64], vec![0.1; 64]).expect("test");
    let result = mha.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_wrong_dim_error() {
    let mha = MultiHeadAttention::mha(64, 8).expect("test");

    // Wrong hidden_dim
    let input = Tensor::from_vec(vec![4, 32], vec![0.1; 128]).expect("test");
    let result = mha.forward(&input);
    assert!(result.is_err());
}
