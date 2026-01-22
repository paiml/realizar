use crate::layers::*;
use crate::layers::*;

#[test]
fn test_layer_norm_creation() {
    let layer_norm = LayerNorm::new(512, 1e-5).expect("test");
    assert_eq!(layer_norm.normalized_shape(), 512);
    assert!((layer_norm.eps() - 1e-5).abs() < 1e-10);
}

#[test]
fn test_layer_norm_zero_shape_error() {
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_forward_simple() {
    // Simple test with known values
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");

    // Input: [1.0, 2.0, 3.0]
    // Mean: 2.0
    // Variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    // Std: sqrt(2/3 + 1e-5) ≈ 0.8165
    // Normalized: [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165]
    //           ≈ [-1.2247, 0.0, 1.2247]
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    let output_data = output.data();
    assert_eq!(output_data.len(), 3);

    // Check that mean is approximately 0
    let mean: f32 = output_data.iter().sum::<f32>() / 3.0;
    assert!((mean - 0.0).abs() < 1e-5);

    // Check that variance is approximately 1
    let variance: f32 = output_data
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / 3.0;
    assert!((variance - 1.0).abs() < 1e-4);
}

#[test]
fn test_layer_norm_forward_batched() {
    // Test with batch dimension
    let layer_norm = LayerNorm::new(2, 1e-5).expect("test");

    // Input: [[1.0, 3.0], [2.0, 4.0]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2]);

    let output_data = output.data();
    assert_eq!(output_data.len(), 4);

    // First group [1.0, 3.0]: mean=2.0, normalized should have mean≈0, var≈1
    let group1_mean = f32::midpoint(output_data[0], output_data[1]);
    assert!((group1_mean - 0.0).abs() < 1e-5);

    // Second group [2.0, 4.0]: mean=3.0, normalized should have mean≈0, var≈1
    let group2_mean = f32::midpoint(output_data[2], output_data[3]);
    assert!((group2_mean - 0.0).abs() < 1e-5);
}

#[test]
fn test_layer_norm_empty_shape_handling() {
    // LayerNorm should handle validation properly
    // Since Tensor itself doesn't allow empty shapes, we test
    // that the normalized_shape validation works
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_shape_mismatch_error() {
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test"); // Wrong size
    let result = layer_norm.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_zero_variance() {
    // Test with constant input (zero variance)
    let layer_norm = LayerNorm::new(3, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![3], vec![2.0, 2.0, 2.0]).expect("test");
    let output = layer_norm.forward(&input).expect("test");

    // With zero variance, normalized values should be close to 0
    // (since eps prevents division by zero)
    let output_data = output.data();
    for &val in output_data {
        assert!(val.abs() < 1e-2); // Should be near 0
    }
}

// Linear layer tests

#[test]
fn test_linear_creation() {
    let linear = Linear::new(128, 256).expect("test");
    assert_eq!(linear.in_features(), 128);
    assert_eq!(linear.out_features(), 256);
}

#[test]
fn test_linear_zero_dimensions_error() {
    let result = Linear::new(0, 256);
    assert!(result.is_err());

    let result = Linear::new(128, 0);
    assert!(result.is_err());
}

#[test]
fn test_linear_forward_simple() {
    // Simple test: 2 in_features, 3 out_features
    let mut linear = Linear::new(2, 3).expect("test");

    // Set identity-like weights for testing
    // weight[i][j] = 1.0 if i==j, 0.0 otherwise (extended for different dimensions)
    linear.weight_mut()[0] = 1.0; // weight[0][0]
    linear.weight_mut()[1] = 0.0; // weight[0][1]
    linear.weight_mut()[2] = 0.0; // weight[0][2]
    linear.weight_mut()[3] = 0.0; // weight[1][0]
    linear.weight_mut()[4] = 1.0; // weight[1][1]
    linear.weight_mut()[5] = 0.0; // weight[1][2]

    // Bias: all 0.5
    linear.bias_mut()[0] = 0.5;
    linear.bias_mut()[1] = 0.5;
    linear.bias_mut()[2] = 0.5;

    // Input: [2.0, 3.0]
    let input = Tensor::from_vec(vec![2], vec![2.0, 3.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[3]);
    let output_data = output.data();

    // Expected: [2.0*1.0 + 3.0*0.0 + 0.5, 2.0*0.0 + 3.0*1.0 + 0.5, 2.0*0.0 + 3.0*0.0 + 0.5]
    //         = [2.5, 3.5, 0.5]
    assert!((output_data[0] - 2.5).abs() < 1e-5);
    assert!((output_data[1] - 3.5).abs() < 1e-5);
    assert!((output_data[2] - 0.5).abs() < 1e-5);
}

#[test]
fn test_linear_forward_batched() {
    // Test with batch dimension: [2, 2] -> [2, 3]
    let mut linear = Linear::new(2, 3).expect("test");

    // Simple weights: all 1.0
    for i in 0..6 {
        linear.weight_mut()[i] = 1.0;
    }
    // Bias: all 0.0
    for i in 0..3 {
        linear.bias_mut()[i] = 0.0;
    }

    // Input: [[1.0, 2.0], [3.0, 4.0]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = linear.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3]);
    let output_data = output.data();

    // First row: [1.0, 2.0] * all-ones weight + zero bias = [3.0, 3.0, 3.0]
    assert!((output_data[0] - 3.0).abs() < 1e-5);
    assert!((output_data[1] - 3.0).abs() < 1e-5);
    assert!((output_data[2] - 3.0).abs() < 1e-5);

    // Second row: [3.0, 4.0] * all-ones weight + zero bias = [7.0, 7.0, 7.0]
    assert!((output_data[3] - 7.0).abs() < 1e-5);
    assert!((output_data[4] - 7.0).abs() < 1e-5);
    assert!((output_data[5] - 7.0).abs() < 1e-5);
}

#[test]
fn test_linear_shape_mismatch_error() {
    let linear = Linear::new(3, 2).expect("test");
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test"); // Wrong size (2 vs 3)
    let result = linear.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_linear_weight_bias_mut() {
    let mut linear = Linear::new(2, 3).expect("test");

    // Modify weights
    linear.weight_mut()[0] = 42.0;
    assert!((linear.weight_mut()[0] - 42.0).abs() < 1e-6);

    // Modify bias
    linear.bias_mut()[0] = 7.0;
    assert!((linear.bias_mut()[0] - 7.0).abs() < 1e-6);
}

// FusedLayerNormLinear tests

#[test]
fn test_fused_layer_norm_linear_creation() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");
    assert_eq!(fused.feature_dim(), 4);
    assert_eq!(fused.out_features(), 8);
}

#[test]
fn test_fused_layer_norm_linear_zero_dims_error() {
    let result = FusedLayerNormLinear::new(0, 8, 1e-5);
    assert!(result.is_err());

    let result = FusedLayerNormLinear::new(4, 0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_matches_separate() {
    // Test that fused implementation matches separate LayerNorm + Linear
    let feature_dim = 4;
    let out_features = 3;

    // Create fused layer
    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set weights
    for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = (i as f32) * 0.1;
        }
    }
    for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = (i as f32) * 0.05;
        }
    }

    // Create separate layers with same weights
    let layer_norm = LayerNorm::new(feature_dim, 1e-5).expect("test");
    let mut linear = Linear::new(feature_dim, out_features).expect("test");
    for (i, weight) in linear.weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = (i as f32) * 0.1;
        }
    }
    for (i, bias) in linear.bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = (i as f32) * 0.05;
        }
    }

    // Test input
    let input = Tensor::from_vec(vec![feature_dim], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    // Fused forward
    let fused_output = fused.forward(&input).expect("test");

    // Separate forward
    let norm_output = layer_norm.forward(&input).expect("test");
    let separate_output = linear.forward(&norm_output).expect("test");

    // Results should match
    assert_eq!(fused_output.shape(), separate_output.shape());
    for i in 0..fused_output.data().len() {
        assert!(
            (fused_output.data()[i] - separate_output.data()[i]).abs() < 1e-4,
            "Mismatch at {}: fused={} vs separate={}",
            i,
            fused_output.data()[i],
            separate_output.data()[i]
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_batched() {
    let feature_dim = 4;
    let out_features = 2;

    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set simple weights
    for weight in fused.linear_weight_mut().iter_mut() {
        *weight = 1.0;
    }

    // Batched input [2, 4]
    let input = Tensor::from_vec(
        vec![2, feature_dim],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .expect("test");

    let output = fused.forward(&input).expect("test");
    assert_eq!(output.shape(), &[2, out_features]);
}

#[test]
fn test_fused_layer_norm_linear_parallel_matches_serial() {
    let feature_dim = 8;
    let out_features = 4;

    let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Set random-ish weights
    for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *weight = ((i * 7 + 3) % 11) as f32 * 0.1;
        }
    }
    for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        {
            *bias = ((i * 5 + 2) % 7) as f32 * 0.1;
        }
    }

    // Large batch
    let mut input_data = Vec::new();
    for i in 0..32 {
        for j in 0..feature_dim {
            #[allow(clippy::cast_precision_loss)]
            {
                input_data.push(((i * feature_dim + j) % 17) as f32 * 0.2);
            }
        }
    }
    let input = Tensor::from_vec(vec![32, feature_dim], input_data).expect("test");

    // Serial
    let serial_output = fused.forward(&input).expect("test");

    // Parallel
    let parallel_output = fused.forward_parallel(&input).expect("test");

    assert_eq!(serial_output.shape(), parallel_output.shape());
    for i in 0..serial_output.data().len() {
        assert!(
            (serial_output.data()[i] - parallel_output.data()[i]).abs() < 1e-5,
            "Mismatch at {}: serial={} vs parallel={}",
            i,
            serial_output.data()[i],
            parallel_output.data()[i]
        );
    }
}

#[test]
fn test_fused_layer_norm_linear_dimension_mismatch_error() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("test");

    // Wrong input dimension
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = fused.forward(&input);
    assert!(result.is_err());
}

// Softmax activation tests

#[test]
fn test_softmax_simple() {
    // Simple softmax: [0, 0, 0] -> [1/3, 1/3, 1/3]
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[3]);
    // All equal inputs -> equal probabilities
    assert!((output.data()[0] - 0.333_333).abs() < 1e-5);
    assert!((output.data()[1] - 0.333_333).abs() < 1e-5);
    assert!((output.data()[2] - 0.333_333).abs() < 1e-5);

    // Sum should be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_probabilities_sum_to_one() {
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Sum should be exactly 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All values should be positive
    for &val in output.data() {
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}

#[test]
fn test_softmax_max_dominates() {
    // When one value is much larger, it should dominate
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 10.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Last element should be close to 1.0
    assert!(output.data()[2] > 0.999);
    // Others should be very small
    assert!(output.data()[0] < 0.001);
    assert!(output.data()[1] < 0.001);
}

#[test]
fn test_softmax_batched() {
    // Batched: [[1, 2], [3, 4]]
    let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[2, 2]);

    // Each row should sum to 1.0
    let row1_sum = output.data()[0] + output.data()[1];
    let row2_sum = output.data()[2] + output.data()[3];
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_numerical_stability() {
    // Large values should not overflow
    let input = Tensor::from_vec(vec![3], vec![1000.0, 1001.0, 1002.0]).expect("test");
    let output = softmax(&input).expect("test");

    // Should not be NaN or Inf
    for &val in output.data() {
        assert!(val.is_finite());
    }

    // Sum should still be 1.0
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_preserves_shape() {
    let input = Tensor::from_vec(vec![2, 3, 4], vec![1.0; 24]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3, 4]);
}

// GELU activation tests

#[test]
fn test_gelu_zero() {
    let input = Tensor::from_vec(vec![1], vec![0.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(0) = 0
    assert!((output.data()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_positive() {
    let input = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(1) ≈ 0.841 (approximately x for large positive x)
    assert!(output.data()[0] > 0.8);
    assert!(output.data()[0] < 0.9);
}

#[test]
fn test_gelu_negative() {
    let input = Tensor::from_vec(vec![1], vec![-1.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(-1) is small negative (smooth near zero)
    assert!(output.data()[0] < 0.0);
    assert!(output.data()[0] > -0.2);
}

#[test]
fn test_gelu_batched() {
    let input =
        Tensor::from_vec(vec![2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).expect("test");
    let output = gelu(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data().len(), 6);

    // GELU(0) = 0
    assert!((output.data()[2] - 0.0).abs() < 1e-6);
    // Positive values should be positive
    assert!(output.data()[3] > 0.0);
    assert!(output.data()[4] > 0.0);
    assert!(output.data()[5] > 0.0);
}

#[test]
fn test_gelu_preserves_shape() {
    // Test that GELU preserves tensor shape
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.5; 24]).expect("test");
    let output = gelu(&input).expect("test");
    assert_eq!(output.shape(), &[2, 3, 4]);
    assert_eq!(output.data().len(), 24);
}

// FeedForward (FFN) tests

#[test]
fn test_ffn_creation() {
    let ffn = FeedForward::new(512, 2048).expect("test");
    assert_eq!(ffn.hidden_dim(), 512);
    assert_eq!(ffn.intermediate_dim(), 2048);
}

#[test]
fn test_ffn_zero_dimensions_error() {
    let result = FeedForward::new(0, 2048);
    assert!(result.is_err());

    let result = FeedForward::new(512, 0);
    assert!(result.is_err());
}

#[test]
fn test_ffn_forward_shape() {
    // Test that FFN preserves hidden_dim
    let ffn = FeedForward::new(4, 16).expect("test"); // Small sizes for testing
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output should have same shape as input
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_ffn_forward_computation() {
    // Test FFN with known weights
    let mut ffn = FeedForward::new(2, 4).expect("test");

    // Set fc1 weights to identity-like (for simplicity)
    for i in 0..8 {
        ffn.fc1_mut().weight_mut()[i] = 0.1;
    }
    for i in 0..4 {
        ffn.fc1_mut().bias_mut()[i] = 0.0;
    }

    // Set fc2 weights
    for i in 0..8 {
        ffn.fc2_mut().weight_mut()[i] = 0.1;
    }
    for i in 0..2 {
        ffn.fc2_mut().bias_mut()[i] = 0.0;
    }

    // Input: [1.0, 2.0]
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output should be valid (not NaN, not Inf)
    assert_eq!(output.shape(), &[2]);
    assert!(output.data()[0].is_finite());
    assert!(output.data()[1].is_finite());
}

#[test]
fn test_ffn_batched() {
    let ffn = FeedForward::new(3, 12).expect("test");

    // Batched input: [2, 3]
    let input = Tensor::from_vec(vec![2, 3], vec![0.5; 6]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output shape should match input
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data().len(), 6);
}

#[test]
fn test_ffn_weight_access() {
    let mut ffn = FeedForward::new(2, 4).expect("test");

    // Modify fc1 weights
    ffn.fc1_mut().weight_mut()[0] = 42.0;
    assert!((ffn.fc1_mut().weight_mut()[0] - 42.0).abs() < 1e-6);

    // Modify fc2 bias
    ffn.fc2_mut().bias_mut()[0] = 7.0;
    assert!((ffn.fc2_mut().bias_mut()[0] - 7.0).abs() < 1e-6);
}

// Attention tests

#[test]
fn test_attention_creation() {
    let attn = Attention::new(64).expect("test");
    assert_eq!(attn.head_dim(), 64);
    // scale = 1 / sqrt(64) = 1/8 = 0.125
    assert!((attn.scale() - 0.125).abs() < 1e-6);
}

#[test]
fn test_attention_zero_head_dim_error() {
    let result = Attention::new(0);
    assert!(result.is_err());
}

#[test]
fn test_attention_forward_shape() {
    let attn = Attention::new(4).expect("test");

    // Q, K, V all have shape [3, 4] (seq_len=3, head_dim=4)
    let q = Tensor::from_vec(vec![3, 4], vec![0.1; 12]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).expect("test");
    let v = Tensor::from_vec(vec![3, 4], vec![0.3; 12]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Output should have shape [3, 4]
    assert_eq!(output.shape(), &[3, 4]);
    assert_eq!(output.data().len(), 12);
}

#[test]
fn test_attention_forward_computation() {
    let attn = Attention::new(2).expect("test");

    // Simple 2x2 case for manual verification
    // Q = [[1, 0], [0, 1]]
    // K = [[1, 0], [0, 1]]
    // V = [[1, 2], [3, 4]]
    let q = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let k = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let v = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Output should be valid (not NaN, not Inf)
    assert_eq!(output.shape(), &[2, 2]);
    for &val in output.data() {
        assert!(val.is_finite());
    }

    // First row of Q=[1,0] has dot products: with K[0]=[1,0] -> 1, with K[1]=[0,1] -> 0
    // After scaling and softmax, should attend more to first position
    // So output[0] should be closer to V[0]=[1,2] than V[1]=[3,4]
    assert!(output.data()[0] < 2.0); // Closer to 1 than 3
    assert!(output.data()[1] < 3.0); // Closer to 2 than 4
}

#[test]
fn test_attention_shape_mismatch_error() {
    let attn = Attention::new(4).expect("test");

    // Q has head_dim=4, K has head_dim=3
    let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![0.2; 6]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_attention_kv_seq_len_mismatch_error() {
    let attn = Attention::new(4).expect("test");

    // K has seq_len=3, V has seq_len=2
    let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_attention_softmax_weights_sum() {
    // Verify that attention is using softmax correctly
    // by checking output is weighted combination of values
    let attn = Attention::new(3).expect("test");

    // All equal Q and K means uniform attention
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    // V = [[1, 2, 3], [4, 5, 6]]
    let v = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // With uniform attention, output should be average of V rows
    // Each output row should be close to [2.5, 3.5, 4.5]
    let expected = [2.5, 3.5, 4.5];
    for row in 0..2 {
        for (col, &exp) in expected.iter().enumerate() {
            let actual = output.data()[row * 3 + col];
            assert!(
                (actual - exp).abs() < 0.01,
                "row={row}, col={col}: expected {exp}, got {actual}",
            );
        }
    }
}

#[test]
fn test_attention_single_position() {
    // Test with single position (seq_len=1)
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // With single position, output should equal V
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

// Flash Attention tests

#[test]
fn test_flash_attention_matches_standard() {
    // Flash Attention should produce same output as standard attention
    let attn = Attention::new(8).expect("test");

    // Create test data
    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data.clone()).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data.clone()).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data.clone()).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with block_size=1 (should be identical)
    let flash_output = attn.flash_forward(&q, &k, &v, 1).expect("test");

    // Results should match
    assert_eq!(standard_output.shape(), flash_output.shape());
    for i in 0..standard_output.data().len() {
        assert!(
            (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-5,
            "Mismatch at index {}: {} vs {}",
            i,
            standard_output.data()[i],
            flash_output.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_multi_position() {
    // Test Flash Attention with multiple positions
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.0, 0.0, 1.0,  // pos 0
        0.0, 1.0, 1.0, 0.0,  // pos 1
        1.0, 1.0, 0.0, 0.0,  // pos 2
    ];
    #[rustfmt::skip]
    let k_data = vec![
        1.0, 0.0, 0.0, 1.0,  // pos 0
        0.0, 1.0, 1.0, 0.0,  // pos 1
        1.0, 1.0, 0.0, 0.0,  // pos 2
    ];
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,  // pos 0
        5.0, 6.0, 7.0, 8.0,  // pos 1
        9.0, 10.0, 11.0, 12.0,  // pos 2
    ];

    let q = Tensor::from_vec(vec![3, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![3, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![3, 4], v_data).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with different block sizes
    for block_size in [1, 2, 3, 4] {
        let flash_output = attn.flash_forward(&q, &k, &v, block_size).expect("test");

        assert_eq!(standard_output.shape(), flash_output.shape());
        for i in 0..standard_output.data().len() {
            assert!(
                (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at index {}: {} vs {}",
                block_size,
                i,
                standard_output.data()[i],
                flash_output.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_large_sequence() {
    // Test with larger sequence to verify block-wise computation
    let attn = Attention::new(8).expect("test");

    // Create larger test data (seq_len=16)
    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..16 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.1);
                k_data.push((i * 8 + j) as f32 * 0.05);
                v_data.push((i * 8 + j) as f32 * 0.2);
            }
        }
    }

    let q = Tensor::from_vec(vec![16, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![16, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![16, 8], v_data).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with block_size=4
    let flash_output = attn.flash_forward(&q, &k, &v, 4).expect("test");

    assert_eq!(standard_output.shape(), flash_output.shape());
    for i in 0..standard_output.data().len() {
        assert!(
            (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-3,
            "Mismatch at index {}: {} vs {}",
            i,
            standard_output.data()[i],
            flash_output.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_shape_errors() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .expect("test");
    let v_wrong = Tensor::from_vec(
        vec![3, 4],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    )
    .expect("test");

    // K/V sequence length mismatch
    let result = attn.flash_forward(&q, &k, &v_wrong, 2);
    assert!(result.is_err());
}

// Flash Attention v2 tests

#[test]
fn test_flash_attention_v2_matches_standard() {
    // Flash Attention v2 with SIMD should match standard attention
    let attn = Attention::new(8).expect("test");

    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 1).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_v2_multi_position() {
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
    ];
    #[rustfmt::skip]
    let k_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
    ];
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];

    let q = Tensor::from_vec(vec![3, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![3, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![3, 4], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");

    for block_size in [1, 2, 3, 4] {
        let v2 = attn.flash_forward_v2(&q, &k, &v, block_size).expect("test");
        assert_eq!(standard.shape(), v2.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at {}: {} vs {}",
                block_size,
                i,
                standard.data()[i],
                v2.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_v2_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_v2_large_sequence() {
    let attn = Attention::new(8).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..32 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.05);
                k_data.push((i * 8 + j) as f32 * 0.03);
                v_data.push((i * 8 + j) as f32 * 0.1);
            }
        }
    }

    let q = Tensor::from_vec(vec![32, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![32, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![32, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 8).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-3,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

// Parallel Flash Attention tests

#[test]
fn test_flash_attention_parallel_matches_standard() {
    let attn = Attention::new(8).expect("test");

    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 1).expect("test");

    assert_eq!(standard.shape(), parallel.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            parallel.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_parallel_multi_position() {
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
        0.7, 0.2, 0.9, 0.5,
    ];
    let k_data = q_data.clone();
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let q = Tensor::from_vec(vec![4, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![4, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![4, 4], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");

    for block_size in [1, 2, 4] {
        let parallel = attn
            .flash_forward_parallel(&q, &k, &v, block_size)
            .expect("test");
        assert_eq!(standard.shape(), parallel.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at {}: {} vs {}",
                block_size,
                i,
                standard.data()[i],
                parallel.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_parallel_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_parallel_large_sequence() {
    let attn = Attention::new(16).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..64 {
        for j in 0..16 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 16 + j) as f32 * 0.02);
                k_data.push((i * 16 + j) as f32 * 0.015);
                v_data.push((i * 16 + j) as f32 * 0.05);
            }
        }
    }

    let q = Tensor::from_vec(vec![64, 16], q_data).expect("test");
    let k = Tensor::from_vec(vec![64, 16], k_data).expect("test");
    let v = Tensor::from_vec(vec![64, 16], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 16).expect("test");

    assert_eq!(standard.shape(), parallel.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - parallel.data()[i]).abs() < 1e-3,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            parallel.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_v2_vs_parallel_consistency() {
    // Both v2 and parallel should produce same results
    let attn = Attention::new(8).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..16 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.1);
                k_data.push((i * 8 + j) as f32 * 0.08);
                v_data.push((i * 8 + j) as f32 * 0.15);
            }
        }
    }

    let q = Tensor::from_vec(vec![16, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![16, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![16, 8], v_data).expect("test");

    let v2 = attn.flash_forward_v2(&q, &k, &v, 4).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 4).expect("test");

    assert_eq!(v2.shape(), parallel.shape());
    for i in 0..v2.data().len() {
        assert!(
            (v2.data()[i] - parallel.data()[i]).abs() < 1e-5,
            "Mismatch at {}: v2={} vs parallel={}",
            i,
            v2.data()[i],
            parallel.data()[i]
        );
    }
}

// RoPE (Rotary Position Embeddings) tests

#[test]
fn test_rope_creation() {
    let rope = RoPE::new(64, 10000.0).expect("test");
    assert_eq!(rope.dim(), 64);
    assert!((rope.base() - 10000.0).abs() < 1e-6);
    assert_eq!(rope.inv_freq().len(), 32); // dim/2
}

#[test]
fn test_rope_with_default_base() {
    let rope = RoPE::with_default_base(128).expect("test");
    assert_eq!(rope.dim(), 128);
    assert!((rope.base() - 10000.0).abs() < 1e-6);
}

#[test]
fn test_rope_zero_dim_error() {
    let result = RoPE::new(0, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_odd_dim_error() {
    let result = RoPE::new(63, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_forward_shape() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let output = rope.forward(&input, 0).expect("test");
    assert_eq!(output.shape(), &[2, 4]);
    assert_eq!(output.data().len(), 8);
}

#[test]
fn test_rope_position_zero_identity() {
    // At position 0, rotation angles are 0, so cos=1, sin=0
    // This should return input unchanged
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = rope.forward(&input, 0).expect("test");

    // At position 0, angles are 0, so cos(0)=1, sin(0)=0
    // y0 = x0 * 1 - x1 * 0 = x0
    // y1 = x0 * 0 + x1 * 1 = x1
    for i in 0..4 {
        assert!(
            (output.data()[i] - input.data()[i]).abs() < 1e-6,
            "Position 0 should be identity: expected {}, got {}",
            input.data()[i],
            output.data()[i]
        );
    }
}

#[test]
fn test_rope_preserves_norm() {
    // Rotation should preserve vector norm
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = rope.forward(&input, 100).expect("test");

    // Compute L2 norm of input pairs and output pairs
    // Each pair should have the same norm after rotation
    let in_norm_0 = (input.data()[0].powi(2) + input.data()[1].powi(2)).sqrt();
    let in_norm_1 = (input.data()[2].powi(2) + input.data()[3].powi(2)).sqrt();
    let out_norm_0 = (output.data()[0].powi(2) + output.data()[1].powi(2)).sqrt();
    let out_norm_1 = (output.data()[2].powi(2) + output.data()[3].powi(2)).sqrt();

    assert!(
        (in_norm_0 - out_norm_0).abs() < 1e-5,
        "Pair 0 norm should be preserved"
    );
    assert!(
        (in_norm_1 - out_norm_1).abs() < 1e-5,
        "Pair 1 norm should be preserved"
    );
}

#[test]
fn test_rope_different_positions() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).expect("test");

    let out_pos_zero = rope.forward(&input, 0).expect("test");
    let out_pos_ten = rope.forward(&input, 10).expect("test");
    let out_pos_hundred = rope.forward(&input, 100).expect("test");

    // Different positions should give different outputs
    assert!(
        (out_pos_zero.data()[0] - out_pos_ten.data()[0]).abs() > 1e-6
            || (out_pos_zero.data()[1] - out_pos_ten.data()[1]).abs() > 1e-6
    );
    assert!(
        (out_pos_ten.data()[0] - out_pos_hundred.data()[0]).abs() > 1e-6
            || (out_pos_ten.data()[1] - out_pos_hundred.data()[1]).abs() > 1e-6
    );
}

#[test]
fn test_rope_dimension_mismatch_error() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![6], vec![1.0; 6]).expect("test");

    let result = rope.forward(&input, 0);
    assert!(result.is_err());
}

#[test]
fn test_rope_batched() {
    // Test with batched input [batch, dim]
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test");

    let output = rope.forward(&input, 5).expect("test");
    assert_eq!(output.shape(), &[3, 4]);

    // All vectors in batch should have same rotation applied
    // (since same position)
    for batch in 0..3 {
        for i in 0..4 {
            let expected = output.data()[i]; // First vector
            let actual = output.data()[batch * 4 + i];
            assert!(
                (expected - actual).abs() < 1e-6,
                "All batch elements should have same rotation"
            );
        }
    }
}

#[test]
fn test_rope_inv_freq_computation() {
    // Test that inverse frequencies are computed correctly
    let rope = RoPE::new(4, 10000.0).expect("test");
    let inv_freq = rope.inv_freq();

    // For dim=4, we have 2 pairs
    // inv_freq[0] = 10000^(-2*0/4) = 10000^0 = 1.0
    // inv_freq[1] = 10000^(-2*1/4) = 10000^(-0.5) = 0.01
    assert!((inv_freq[0] - 1.0).abs() < 1e-6);
    assert!((inv_freq[1] - 0.01).abs() < 1e-6);
}

// ScaledRoPE (NTK, YaRN, Linear, Dynamic NTK) tests

#[test]
fn test_scaled_rope_no_scaling() {
    // ScaledRoPE with None scaling should behave like regular RoPE
    let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("test");
    assert_eq!(scaled.dim(), 64);
    assert!((scaled.original_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.scaled_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    assert!((scaled.context_length_multiplier() - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_linear_scaling() {
    // Linear scaling (Code Llama style)
    let scaling = RopeScalingType::Linear { scale: 4.0 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // Linear scaling doesn't change base frequency
    assert!((scaled.scaled_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_ntk_scaling() {
    // NTK-aware scaling
    let scaling = RopeScalingType::Ntk { scale: 4.0 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // NTK should increase base: base' = base * scale^(dim/(dim-2))
    // For dim=64: exponent = 64/62 ≈ 1.032
    // scaled_base = 10000 * 4^1.032 ≈ 41,376
    assert!(scaled.scaled_base() > 10000.0);
    assert!(scaled.scaled_base() > 40000.0);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_dynamic_ntk() {
    // Dynamic NTK scaling
    let scaling = RopeScalingType::DynamicNtk {
        original_max_len: 2048,
        target_max_len: 8192,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    // Should behave like NTK with scale = 4.0
    assert!(scaled.scaled_base() > 40000.0);
}

#[test]
fn test_scaled_rope_yarn() {
    // YaRN scaling
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 32768,
        attn_factor: 0.0, // Compute automatically
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    // Context multiplier = 32768 / 2048 = 16
    assert!((scaled.context_length_multiplier() - 16.0).abs() < 1e-6);
    // YaRN should have mscale > 1.0 for large extensions
    assert!(scaled.mscale() > 1.0);
    // YaRN should have modified base
    assert!(scaled.scaled_base() > 10000.0);
}

#[test]
fn test_scaled_rope_yarn_custom_attn_factor() {
    // YaRN with custom attention factor
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.5, // Custom value
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("test");

    // Should use custom attn_factor
    assert!((scaled.mscale() - 1.5).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_forward_no_scaling() {
    let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let output = scaled.forward(&input, 0).expect("test");

    // At position 0, rotation should be identity-like
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_forward_linear() {
    let scaling = RopeScalingType::Linear { scale: 2.0 };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    // Position 10 with scale 2 should behave like position 5
    let output = scaled.forward(&input, 10).expect("test");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_forward_ntk() {
    let scaling = RopeScalingType::Ntk { scale: 4.0 };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    let output = scaled.forward(&input, 100).expect("test");
    assert_eq!(output.shape(), &[4]);
    // Output should preserve norm (rotation is norm-preserving)
    let norm: f32 = output.data().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 2.0_f32.sqrt()).abs() < 0.1);
}

#[test]
fn test_scaled_rope_forward_yarn() {
    let scaling = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let scaled = ScaledRoPE::new(4, 10000.0, scaling).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 1.0]).expect("test");

    let output = scaled.forward(&input, 5000).expect("test");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_scaled_rope_zero_dim_error() {
    let result = ScaledRoPE::new(0, 10000.0, RopeScalingType::None);
    assert!(result.is_err());
}

#[test]
fn test_scaled_rope_odd_dim_error() {
    let result = ScaledRoPE::new(63, 10000.0, RopeScalingType::None);
    assert!(result.is_err());
}

#[test]
fn test_scaled_rope_dimension_mismatch() {
    let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("test");
    let input = Tensor::from_vec(vec![8], vec![0.0; 8]).expect("test");

    let result = scaled.forward(&input, 0);
    assert!(result.is_err());
}

#[test]
fn test_rope_scaling_type_default() {
    let scaling = RopeScalingType::default();
    assert_eq!(scaling, RopeScalingType::None);
}

#[test]
fn test_scaled_rope_with_default_base() {
    let scaled = ScaledRoPE::with_default_base(64, RopeScalingType::None).expect("test");
    assert!((scaled.original_base() - 10000.0).abs() < 1e-6);
}

#[test]
fn test_scaled_rope_inv_freq_length() {
    let scaled = ScaledRoPE::new(128, 10000.0, RopeScalingType::None).expect("test");
    assert_eq!(scaled.inv_freq().len(), 64); // dim / 2
}

// ALiBi (Attention with Linear Biases) tests

#[test]
fn test_alibi_creation() {
    let alibi = ALiBi::new(8).expect("test");
    assert_eq!(alibi.num_heads(), 8);
    assert_eq!(alibi.slopes().len(), 8);
}

#[test]
fn test_alibi_zero_heads_error() {
    let result = ALiBi::new(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_slopes_power_of_2() {
    // For 8 heads (power of 2), slopes should follow: 2^(-8h/8) = 2^(-h)
    let alibi = ALiBi::new(8).expect("test");
    let slopes = alibi.slopes();

    // Expected slopes: 2^0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7
    assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0 = 1.0
    assert!((slopes[1] - 0.5).abs() < 1e-6); // 2^-1 = 0.5
    assert!((slopes[2] - 0.25).abs() < 1e-6); // 2^-2 = 0.25
    assert!((slopes[3] - 0.125).abs() < 1e-6); // 2^-3 = 0.125
}

#[test]
fn test_alibi_slopes_non_power_of_2() {
    // For 6 heads (not power of 2)
    let alibi = ALiBi::new(6).expect("test");
    let slopes = alibi.slopes();

    assert_eq!(slopes.len(), 6);

    // First 4 slopes follow 2^(-8h/4) = 2^(-2h)
    assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0
    assert!((slopes[1] - 0.25).abs() < 1e-6); // 2^-2
    assert!((slopes[2] - 0.0625).abs() < 1e-6); // 2^-4
    assert!((slopes[3] - 0.015_625).abs() < 1e-6); // 2^-6

    // Extra 2 slopes follow 2^(-4h/4) with step=2
    // slopes[4] = 2^(-1) = 0.5
    // slopes[5] = 2^(-3) = 0.125
    assert!((slopes[4] - 0.5).abs() < 1e-6);
    assert!((slopes[5] - 0.125).abs() < 1e-6);
}

#[test]
fn test_alibi_bias_shape() {
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(10).expect("test");

    // Shape should be [seq_len, seq_len, num_heads]
    assert_eq!(bias.shape(), &[10, 10, 4]);
}

#[test]
fn test_alibi_bias_zero_seq_len_error() {
    let alibi = ALiBi::new(4).expect("test");
    let result = alibi.get_bias(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_bias_diagonal_zero() {
    // Diagonal elements (same position) should be zero
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(5).expect("test");

    for i in 0..5 {
        for h in 0..4 {
            let idx = i * 5 * 4 + i * 4 + h; // [i, i, h]
            let value = bias.data()[idx];
            assert!(
                value.abs() < 1e-6,
                "Diagonal bias[{i}, {i}, {h}] should be 0, got {value}"
            );
        }
    }
}

#[test]
fn test_alibi_bias_symmetry() {
    // |i - j| = |j - i|, so bias[i,j,h] should equal bias[j,i,h]
    let alibi = ALiBi::new(2).expect("test");
    let bias = alibi.get_bias(4).expect("test");

    for i in 0..4 {
        for j in 0..4 {
            for h in 0..2 {
                let idx_ij = i * 4 * 2 + j * 2 + h;
                let idx_ji = j * 4 * 2 + i * 2 + h;
                let bias_ij = bias.data()[idx_ij];
                let bias_ji = bias.data()[idx_ji];
                assert!(
                    (bias_ij - bias_ji).abs() < 1e-6,
                    "Bias should be symmetric: [{i},{j},{h}]={bias_ij} vs [{j},{i},{h}]={bias_ji}"
                );
            }
        }
    }
}

#[test]
fn test_alibi_bias_computation() {
    // Test exact bias values
    let alibi = ALiBi::new(2).expect("test");
    let slopes = alibi.slopes();
    let bias = alibi.get_bias(3).expect("test");

    // For 2 heads: slopes = [1.0, 0.0625]
    // bias[0, 2, 0] = -slopes[0] * |0 - 2| = -1.0 * 2 = -2.0
    let idx = 2 * 2;
    assert!(
        (bias.data()[idx] - (-2.0)).abs() < 1e-6,
        "Expected -2.0, got {}",
        bias.data()[idx]
    );

    // bias[1, 2, 1] = -slopes[1] * |1 - 2| = -0.0625 * 1 = -0.0625
    let idx = 3 * 2 + 2 * 2 + 1;
    let expected = -slopes[1];
    assert!(
        (bias.data()[idx] - expected).abs() < 1e-6,
        "Expected {expected}, got {}",
        bias.data()[idx]
    );
}

#[test]
fn test_alibi_bias_negative() {
    // All bias values should be <= 0 (except diagonal which is 0)
    let alibi = ALiBi::new(4).expect("test");
    let bias = alibi.get_bias(10).expect("test");

    for &value in bias.data() {
        assert!(value <= 1e-6, "Bias should be non-positive, got {value}");
    }
}

#[test]
fn test_alibi_bias_distance_proportional() {
    // Bias should be proportional to distance
    let alibi = ALiBi::new(1).expect("test");
    let bias = alibi.get_bias(5).expect("test");

    // For head 0, slope is 1.0
    // bias[0, 1] = -1.0 * 1 = -1.0
    // bias[0, 2] = -1.0 * 2 = -2.0
    // bias[0, 3] = -1.0 * 3 = -3.0

    let bias_01 = bias.data()[1];
    let bias_02 = bias.data()[2];
    let bias_03 = bias.data()[3];

    assert!((bias_01 - (-1.0)).abs() < 1e-6);
    assert!((bias_02 - (-2.0)).abs() < 1e-6);
    assert!((bias_03 - (-3.0)).abs() < 1e-6);
}

#[test]
fn test_alibi_single_head() {
    let alibi = ALiBi::new(1).expect("test");
    assert_eq!(alibi.num_heads(), 1);
    assert_eq!(alibi.slopes().len(), 1);
    assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6); // First slope is 2^0 = 1.0
}

#[test]
fn test_alibi_large_num_heads() {
    // Test with large number of heads (non-power of 2)
    let alibi = ALiBi::new(12).expect("test");
    assert_eq!(alibi.num_heads(), 12);
    assert_eq!(alibi.slopes().len(), 12);

    // All slopes should be positive
    for slope in alibi.slopes() {
        assert!(*slope > 0.0, "Slope should be positive, got {slope}");
    }

    // First head should have largest slope (1.0)
    assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_alibi_bias_long_sequence() {
    // Test with longer sequence
    let alibi = ALiBi::new(8).expect("test");
    let bias = alibi.get_bias(128).expect("test");

    assert_eq!(bias.shape(), &[128, 128, 8]);

    // Check that far positions have larger negative bias
    let near_bias = bias.data()[8]; // distance 1
    let far_bias = bias.data()[100 * 8]; // distance 100

    assert!(near_bias > far_bias); // near should be less negative
}

// KVCache tests

