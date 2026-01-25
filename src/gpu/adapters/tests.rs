//! Phase 42 - Adapter Coverage Tests
//!
//! Tests CPU-only logic in adapters without requiring CUDA.

use super::apr::{transpose_matrix, AprGpuError, AprToGpuAdapter};
use crate::apr_transformer::{AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4};

// ============================================================================
// AprGpuError Tests
// ============================================================================

#[test]
fn test_apr_gpu_error_dequant() {
    let err = AprGpuError::DequantError("test error".to_string());
    assert!(err.to_string().contains("dequantize"));
    assert!(err.to_string().contains("test error"));
}

#[test]
fn test_apr_gpu_error_dimension_mismatch() {
    let err = AprGpuError::DimensionMismatch {
        expected: 100,
        actual: 50,
    };
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("50"));
}

#[test]
fn test_apr_gpu_error_gpu_model() {
    let err = AprGpuError::GpuModelError("allocation failed".to_string());
    assert!(err.to_string().contains("GpuModel"));
    assert!(err.to_string().contains("allocation failed"));
}

#[test]
fn test_apr_gpu_error_debug() {
    let err = AprGpuError::DequantError("test".to_string());
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("DequantError"));
}

// ============================================================================
// transpose_matrix Tests (Phase 42)
// ============================================================================

#[test]
fn test_transpose_square() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 2, 2);
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_rectangular() {
    // 2x3 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_matrix(&data, 2, 3);
    // Becomes 3x2
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_tall() {
    // 3x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_matrix(&data, 3, 2);
    // Becomes 2x3
    assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_transpose_single_row() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 1, 4);
    // 1x4 becomes 4x1
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_single_col() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 4, 1);
    // 4x1 becomes 1x4
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_identity() {
    let data = vec![1.0, 0.0, 0.0, 1.0];
    let result = transpose_matrix(&data, 2, 2);
    // Identity transposes to itself
    assert_eq!(result, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_transpose_large() {
    // 4x8 matrix
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let result = transpose_matrix(&data, 4, 8);

    // Verify dimensions preserved (8x4)
    assert_eq!(result.len(), 32);

    // Verify specific elements: (0,0) -> (0,0), (0,7) -> (7,0), (3,0) -> (0,3)
    assert_eq!(result[0], 0.0); // (0,0)
    assert_eq!(result[4 * 7], 7.0); // (7,0) was (0,7)
    assert_eq!(result[3], 24.0); // (0,3) was (3,0)
}

#[test]
fn test_transpose_empty() {
    let data: Vec<f32> = vec![];
    let result = transpose_matrix(&data, 0, 0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_transpose_single_element() {
    let data = vec![42.0];
    let result = transpose_matrix(&data, 1, 1);
    assert_eq!(result, vec![42.0]);
}

// ============================================================================
// AprToGpuAdapter Config Tests
// ============================================================================

#[test]
fn test_config_to_gpu_all_fields() {
    let apr_config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128256,
        intermediate_dim: 14336,
        context_length: 4096,
        rope_theta: 500000.0,
        eps: 1e-6,
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.vocab_size, 128256);
    assert_eq!(gpu_config.hidden_dim, 4096);
    assert_eq!(gpu_config.num_heads, 32);
    assert_eq!(gpu_config.num_kv_heads, 8);
    assert_eq!(gpu_config.num_layers, 32);
    assert_eq!(gpu_config.intermediate_dim, 14336);
    assert_eq!(gpu_config.eps, 1e-6);
    assert_eq!(gpu_config.rope_theta, 500000.0);
}

#[test]
fn test_config_to_gpu_gqa() {
    // GQA: num_kv_heads < num_heads
    let apr_config = AprTransformerConfig {
        architecture: "mistral".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        vocab_size: 32000,
        intermediate_dim: 14336,
        context_length: 32768,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    // GQA should be preserved
    assert_eq!(gpu_config.num_heads, 32);
    assert_eq!(gpu_config.num_kv_heads, 8);
}

#[test]
fn test_config_to_gpu_mha() {
    // MHA: num_kv_heads == num_heads
    let apr_config = AprTransformerConfig {
        architecture: "gpt2".to_string(),
        hidden_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12, // MHA
        vocab_size: 50257,
        intermediate_dim: 3072,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.num_heads, gpu_config.num_kv_heads);
}

#[test]
fn test_config_to_gpu_tiny_model() {
    let apr_config = AprTransformerConfig {
        architecture: "tiny".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.hidden_dim, 64);
    assert_eq!(gpu_config.vocab_size, 100);
}

// ============================================================================
// AprToGpuAdapter Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_tensor_empty() {
    let result = AprToGpuAdapter::dequantize_tensor(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dequantize_tensor_padding() {
    // Q4_0 block: 2 bytes scale + 16 bytes nibbles = 18 bytes for 32 values
    // Create minimal valid Q4_0 data
    let mut data = vec![0u8; 18];
    // Set scale to 1.0 (f16)
    data[0] = 0x00;
    data[1] = 0x3c; // f16 1.0

    let result = AprToGpuAdapter::dequantize_tensor(&data, 64);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 64); // Padded to expected
}

#[test]
fn test_dequantize_tensor_truncation() {
    // Create 2 Q4_0 blocks (64 values)
    let mut data = vec![0u8; 36];
    data[0] = 0x00;
    data[1] = 0x3c;
    data[18] = 0x00;
    data[19] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 32); // Truncated to expected
}

#[test]
fn test_dequantize_tensor_exact_size() {
    // Create 1 Q4_0 block (32 values)
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 32);
}

// ============================================================================
// AprToGpuAdapter Weight Extraction Tests
// ============================================================================

#[test]
fn test_extract_qkv_weights_dimensions() {
    let layer = create_test_q4_layer(256, 4, 4, 512);

    // QKV: hidden_dim + 2 * kv_dim = 256 + 2*256 = 768
    // Total: hidden_dim * qkv_out = 256 * 768 = 196608
    let result = AprToGpuAdapter::extract_qkv_weights(&layer, 256, 4, 4);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), 256 * 768);
}

#[test]
fn test_extract_qkv_weights_gqa() {
    let layer = create_test_q4_layer(256, 8, 2, 512); // GQA: 8 heads, 2 kv heads

    // QKV: hidden_dim + 2 * kv_dim = 256 + 2*64 = 384
    let result = AprToGpuAdapter::extract_qkv_weights(&layer, 256, 8, 2);
    assert!(result.is_ok());

    let weights = result.unwrap();
    // hidden_dim * qkv_out = 256 * 384 = 98304
    assert_eq!(weights.len(), 256 * 384);
}

#[test]
fn test_extract_out_weights() {
    let layer = create_test_q4_layer(256, 4, 4, 512);

    let result = AprToGpuAdapter::extract_out_weights(&layer, 256);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), 256 * 256);
}

#[test]
fn test_extract_ffn_weights() {
    let layer = create_test_q4_layer(256, 4, 4, 512);

    let result = AprToGpuAdapter::extract_ffn_weights(&layer, 256, 512);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), 256 * 512); // up projection
    assert_eq!(fc2.len(), 512 * 256); // down projection
}

#[test]
fn test_extract_ffn_weights_large() {
    let layer = create_test_q4_layer(1024, 16, 16, 4096);

    let result = AprToGpuAdapter::extract_ffn_weights(&layer, 1024, 4096);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), 1024 * 4096);
    assert_eq!(fc2.len(), 4096 * 1024);
}

// ============================================================================
// GpuModelQ4 CPU Logic Tests (Phase 42)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_model_q4_tests {
    use super::*;
    use crate::apr_transformer::QuantizedAprTransformerQ4;
    use crate::gpu::adapters::apr_q4::{AprQ4ToGpuAdapter, GpuModelQ4, LayerNorms};

    fn create_test_gpu_model_q4() -> GpuModelQ4 {
        GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 1000,
                intermediate_dim: 128,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.1; 1000 * 64],
            output_norm_weight: vec![1.0; 64],
            layer_norms: vec![
                LayerNorms {
                    attn_norm: vec![1.0; 64],
                    ffn_norm: vec![1.0; 64],
                },
                LayerNorms {
                    attn_norm: vec![1.0; 64],
                    ffn_norm: vec![1.0; 64],
                },
            ],
            num_layers: 2,
            has_gate: true,
        }
    }

    #[test]
    fn test_layer_norms_clone() {
        let norms = LayerNorms {
            attn_norm: vec![1.0, 2.0, 3.0],
            ffn_norm: vec![4.0, 5.0, 6.0],
        };

        let cloned = norms.clone();

        assert_eq!(cloned.attn_norm, norms.attn_norm);
        assert_eq!(cloned.ffn_norm, norms.ffn_norm);
    }

    #[test]
    fn test_layer_norms_debug() {
        let norms = LayerNorms {
            attn_norm: vec![1.0],
            ffn_norm: vec![2.0],
        };

        let debug_str = format!("{:?}", norms);
        assert!(debug_str.contains("LayerNorms"));
    }

    #[test]
    fn test_gpu_model_q4_config() {
        let model = create_test_gpu_model_q4();
        assert_eq!(model.num_layers, 2);
        assert!(model.has_gate);
        assert_eq!(model.config.hidden_dim, 64);
    }

    #[test]
    fn test_gpu_model_q4_clone() {
        let model = create_test_gpu_model_q4();
        let cloned = model.clone();

        assert_eq!(cloned.num_layers, model.num_layers);
        assert_eq!(cloned.has_gate, model.has_gate);
        assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
    }

    #[test]
    fn test_gpu_model_q4_debug() {
        let model = create_test_gpu_model_q4();
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("GpuModelQ4"));
    }

    #[test]
    fn test_create_model_with_gate() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert!(model.has_gate);
        assert_eq!(model.num_layers, 2);
    }

    #[test]
    fn test_create_model_without_gate() {
        let apr = create_test_quantized_apr(false);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert!(!model.has_gate);
    }

    #[test]
    fn test_create_model_layer_norms_extracted() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.layer_norms.len(), 2);
        assert_eq!(model.layer_norms[0].attn_norm.len(), 64);
        assert_eq!(model.layer_norms[0].ffn_norm.len(), 64);
    }

    #[test]
    fn test_create_model_embeddings_copied() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.token_embedding.len(), apr.token_embedding.len());
        assert_eq!(model.output_norm_weight.len(), apr.output_norm_weight.len());
    }

    #[test]
    fn test_create_model_config_preserved() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.config.hidden_dim, apr.config.hidden_dim);
        assert_eq!(model.config.num_heads, apr.config.num_heads);
        assert_eq!(model.config.num_kv_heads, apr.config.num_kv_heads);
        assert_eq!(model.config.vocab_size, apr.config.vocab_size);
    }

    #[test]
    fn test_create_model_empty_layers() {
        let apr = QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 64,
                num_layers: 0,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 100,
                intermediate_dim: 128,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.1; 100 * 64],
            layers: vec![],
            output_norm_weight: vec![1.0; 64],
            lm_head_weight: QuantizedAprTensorQ4::zeros(64, 100),
        };

        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.num_layers, 0);
        assert!(!model.has_gate);
        assert!(model.layer_norms.is_empty());
    }

    fn create_test_quantized_apr(with_gate: bool) -> QuantizedAprTransformerQ4 {
        let hidden_dim = 64;
        let intermediate_dim = 128;
        let vocab_size = 100;

        QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size,
                intermediate_dim,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers: vec![
                create_test_q4_layer_full(hidden_dim, 4, 4, intermediate_dim, with_gate),
                create_test_q4_layer_full(hidden_dim, 4, 4, intermediate_dim, with_gate),
            ],
            output_norm_weight: vec![1.0; hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
        }
    }

    fn create_test_q4_layer_full(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        with_gate: bool,
    ) -> QuantizedAprLayerQ4 {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: if with_gate {
                Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
            } else {
                None
            },
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_q4_layer(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
) -> QuantizedAprLayerQ4 {
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}
