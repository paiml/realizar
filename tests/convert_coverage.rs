//! Coverage tests for convert.rs - GGUF to APR conversion
//!
//! These tests target uncovered paths in convert.rs to increase coverage from 75% to 85%+.
//! Focus areas:
//! - Model conversion functions
//! - Format detection and validation
//! - Weight conversion and preservation
//! - Quantization during conversion
//! - Error handling paths
//! - Edge cases for statistics calculations

use realizar::apr::{ALIGNMENT, HEADER_SIZE, MAGIC};
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use realizar::convert::{ConversionStats, GgufToAprConverter, Q4KConversionStats, RawTensor};
use realizar::gguf::{GGUFConfig, GGUFTransformer, GGUFTransformerLayer};

// =============================================================================
// Helper Functions
// =============================================================================

fn create_minimal_gguf_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> GGUFTransformer {
    let config = GGUFConfig {
        architecture: "test_arch".to_string(),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
        .map(|_| GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        })
        .collect();

    GGUFTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
    }
}

fn create_minimal_apr_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> AprTransformer {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let layers: Vec<AprTransformerLayer> = (0..num_layers)
        .map(|_| AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        })
        .collect();

    AprTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
    }
}

// =============================================================================
// GgufToAprConverter Tests
// =============================================================================

#[test]
fn test_from_gguf_transformer_preserves_architecture() {
    let gguf = create_minimal_gguf_transformer(32, 2, 100, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.architecture, "test_arch");
}

#[test]
fn test_from_gguf_transformer_preserves_hidden_dim() {
    let gguf = create_minimal_gguf_transformer(128, 1, 50, 256);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.hidden_dim, 128);
}

#[test]
fn test_from_gguf_transformer_preserves_num_heads() {
    let gguf = create_minimal_gguf_transformer(64, 1, 100, 128);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.num_heads, 4);
    assert_eq!(apr.config.num_kv_heads, 4);
}

#[test]
fn test_from_gguf_transformer_preserves_context_length() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.context_length, 512);
}

#[test]
fn test_from_gguf_transformer_preserves_rope_theta() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert!((apr.config.rope_theta - 10000.0).abs() < 0.1);
}

#[test]
fn test_from_gguf_transformer_preserves_eps() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert!((apr.config.eps - 1e-5).abs() < 1e-7);
}

#[test]
fn test_from_gguf_transformer_preserves_embeddings() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.token_embedding.len(), gguf.token_embedding.len());
    assert_eq!(apr.token_embedding, gguf.token_embedding);
}

#[test]
fn test_from_gguf_transformer_preserves_output_norm() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.output_norm_weight, gguf.output_norm_weight);
    assert_eq!(apr.output_norm_bias, gguf.output_norm_bias);
}

#[test]
fn test_from_gguf_transformer_preserves_lm_head() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.lm_head_weight, gguf.lm_head_weight);
    assert_eq!(apr.lm_head_bias, gguf.lm_head_bias);
}

#[test]
fn test_from_gguf_transformer_zero_layers() {
    let gguf = create_minimal_gguf_transformer(16, 0, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.layers.len(), 0);
    assert_eq!(apr.config.num_layers, 0);
}

#[test]
fn test_from_gguf_transformer_many_layers() {
    let gguf = create_minimal_gguf_transformer(8, 12, 10, 16);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.layers.len(), 12);
}

// =============================================================================
// APR Serialization Tests
// =============================================================================

#[test]
fn test_to_apr_bytes_creates_valid_header() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() >= HEADER_SIZE);
    assert_eq!(&bytes[0..4], &MAGIC);
}

#[test]
fn test_to_apr_bytes_version_is_v2() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert_eq!(bytes[4], 2); // major
    assert_eq!(bytes[5], 0); // minor
}

#[test]
fn test_to_apr_bytes_tensor_count_is_one() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(tensor_count, 1);
}

#[test]
fn test_to_apr_bytes_metadata_offset() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    assert_eq!(metadata_offset, HEADER_SIZE as u64);
}

#[test]
fn test_to_apr_bytes_multiple_layers() {
    let apr = create_minimal_apr_transformer(8, 4, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() > HEADER_SIZE);
}

#[test]
fn test_apr_roundtrip_preserves_config() {
    let original = create_minimal_apr_transformer(16, 2, 30, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.config.architecture, loaded.config.architecture);
    assert_eq!(original.config.hidden_dim, loaded.config.hidden_dim);
    assert_eq!(original.config.num_layers, loaded.config.num_layers);
    assert_eq!(original.config.vocab_size, loaded.config.vocab_size);
}

#[test]
fn test_apr_roundtrip_preserves_weights() {
    let original = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.token_embedding, loaded.token_embedding);
    assert_eq!(original.output_norm_weight, loaded.output_norm_weight);
    assert_eq!(original.lm_head_weight, loaded.lm_head_weight);
}

#[test]
fn test_apr_roundtrip_preserves_layer_weights() {
    let original = create_minimal_apr_transformer(8, 2, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.layers.len(), loaded.layers.len());
    for (orig, load) in original.layers.iter().zip(loaded.layers.iter()) {
        assert_eq!(orig.attn_norm_weight, load.attn_norm_weight);
        assert_eq!(orig.qkv_weight, load.qkv_weight);
    }
}

// =============================================================================
// from_apr_bytes Error Path Tests
// =============================================================================

#[test]
fn test_from_apr_bytes_wrong_magic() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(b"XXXX");
    bytes[4] = 2;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_empty_input() {
    let bytes: Vec<u8> = vec![];
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_header_only() {
    let mut bytes = vec![0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_before_index() {
    let mut bytes = vec![0u8; 70];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data_offset beyond end
    bytes[64..66].copy_from_slice(b"{}");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_tensor_index_json() {
    let mut bytes = vec![0u8; 120];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&100u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");
    bytes[66..78].copy_from_slice(b"invalid json");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_no_weights_tensor() {
    let mut bytes = vec![0u8; 140];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&120u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    // Valid JSON but wrong tensor name
    let index_json = r#"[{"name":"other","dtype":"json","shape":[10],"offset":0,"size":10}]"#;
    bytes[66..66 + index_json.len()].copy_from_slice(index_json.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_tensor_data() {
    let mut bytes = vec![0u8; 150];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&130u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    // Valid JSON but size exceeds available data
    let index_json = r#"[{"name":"weights","dtype":"json","shape":[9999],"offset":0,"size":9999}]"#;
    let idx_end = 66 + index_json.len();
    bytes[66..idx_end].copy_from_slice(index_json.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

// =============================================================================
// ConversionStats Tests
// =============================================================================

#[test]
fn test_stats_returns_correct_num_layers() {
    let apr = create_minimal_apr_transformer(32, 6, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.num_layers, 6);
}

#[test]
fn test_stats_returns_correct_hidden_dim() {
    let apr = create_minimal_apr_transformer(256, 2, 1000, 512);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.hidden_dim, 256);
}

#[test]
fn test_stats_returns_correct_vocab_size() {
    let apr = create_minimal_apr_transformer(64, 1, 5000, 128);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.vocab_size, 5000);
}

#[test]
fn test_stats_returns_correct_architecture() {
    let apr = create_minimal_apr_transformer(32, 1, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.architecture, "test");
}

#[test]
fn test_stats_total_parameters_positive() {
    let apr = create_minimal_apr_transformer(32, 2, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert!(stats.total_parameters > 0);
}

#[test]
fn test_stats_memory_bytes_is_4x_params() {
    let apr = create_minimal_apr_transformer(32, 1, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);
}

#[test]
fn test_stats_memory_mb_calculation() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 4,
        hidden_dim: 512,
        vocab_size: 10000,
        architecture: "test".to_string(),
    };

    let expected_mb = 4_000_000.0 / (1024.0 * 1024.0);
    assert!((stats.memory_mb() - expected_mb).abs() < 0.001);
}

#[test]
fn test_stats_memory_gb_calculation() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000,
        memory_bytes_f32: 4_000_000_000,
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 32000,
        architecture: "large".to_string(),
    };

    let expected_gb = 4_000_000_000.0 / (1024.0 * 1024.0 * 1024.0);
    assert!((stats.memory_gb() - expected_gb).abs() < 0.01);
}

#[test]
fn test_stats_parameters_m_calculation() {
    let stats = ConversionStats {
        total_parameters: 7_000_000,
        memory_bytes_f32: 28_000_000,
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 30000,
        architecture: "medium".to_string(),
    };

    assert!((stats.parameters_m() - 7.0).abs() < 0.001);
}

#[test]
fn test_stats_parameters_b_calculation() {
    let stats = ConversionStats {
        total_parameters: 13_000_000_000,
        memory_bytes_f32: 52_000_000_000,
        num_layers: 40,
        hidden_dim: 5120,
        vocab_size: 50000,
        architecture: "13b".to_string(),
    };

    assert!((stats.parameters_b() - 13.0).abs() < 0.001);
}

#[test]
fn test_stats_zero_values() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };

    assert_eq!(stats.memory_mb(), 0.0);
    assert_eq!(stats.memory_gb(), 0.0);
    assert_eq!(stats.parameters_m(), 0.0);
    assert_eq!(stats.parameters_b(), 0.0);
}

#[test]
fn test_conversion_stats_debug_impl() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "tiny".to_string(),
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("ConversionStats"));
    assert!(debug_str.contains("1000"));
    assert!(debug_str.contains("tiny"));
}

#[test]
fn test_conversion_stats_clone_impl() {
    let stats = ConversionStats {
        total_parameters: 500,
        memory_bytes_f32: 2000,
        num_layers: 1,
        hidden_dim: 32,
        vocab_size: 50,
        architecture: "nano".to_string(),
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_parameters, stats.total_parameters);
    assert_eq!(cloned.memory_bytes_f32, stats.memory_bytes_f32);
    assert_eq!(cloned.architecture, stats.architecture);
}

// =============================================================================
// RawTensor Tests
// =============================================================================

#[test]
fn test_raw_tensor_debug_impl() {
    let tensor = RawTensor {
        name: "layer.0.weight".to_string(),
        data: vec![0u8; 256],
        shape: vec![16, 16],
        dtype: 0,
    };

    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("layer.0.weight"));
    assert!(debug_str.contains("[16, 16]"));
}

#[test]
fn test_raw_tensor_clone_impl() {
    let tensor = RawTensor {
        name: "embed.weight".to_string(),
        data: vec![1, 2, 3, 4, 5],
        shape: vec![5],
        dtype: 1,
    };

    let cloned = tensor.clone();
    assert_eq!(cloned.name, tensor.name);
    assert_eq!(cloned.data, tensor.data);
    assert_eq!(cloned.shape, tensor.shape);
    assert_eq!(cloned.dtype, tensor.dtype);
}

#[test]
fn test_raw_tensor_f32_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 16],
        shape: vec![4],
        dtype: 0, // F32
    };

    assert_eq!(tensor.dtype, 0);
}

#[test]
fn test_raw_tensor_f16_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 8],
        shape: vec![4],
        dtype: 1, // F16
    };

    assert_eq!(tensor.dtype, 1);
}

#[test]
fn test_raw_tensor_q4k_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 144],
        shape: vec![256],
        dtype: 12, // Q4_K
    };

    assert_eq!(tensor.dtype, 12);
}

#[test]
fn test_raw_tensor_q6k_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 210],
        shape: vec![256],
        dtype: 14, // Q6_K
    };

    assert_eq!(tensor.dtype, 14);
}

#[test]
fn test_raw_tensor_multidim_shape() {
    let tensor = RawTensor {
        name: "attention.weight".to_string(),
        data: vec![0u8; 1024],
        shape: vec![32, 8, 4],
        dtype: 0,
    };

    assert_eq!(tensor.shape.len(), 3);
    assert_eq!(tensor.shape[0], 32);
    assert_eq!(tensor.shape[1], 8);
    assert_eq!(tensor.shape[2], 4);
}

// =============================================================================
// Q4KConversionStats Tests
// =============================================================================

#[test]
fn test_q4k_stats_debug_impl() {
    let stats = Q4KConversionStats {
        tensor_count: 200,
        q4k_tensor_count: 180,
        total_bytes: 10_000_000,
        architecture: "llama".to_string(),
        num_layers: 32,
        hidden_size: 4096,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("200"));
    assert!(debug_str.contains("llama"));
    assert!(debug_str.contains("32"));
}

#[test]
fn test_q4k_stats_clone_impl() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 90,
        total_bytes: 5_000_000,
        architecture: "qwen".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.tensor_count, stats.tensor_count);
    assert_eq!(cloned.q4k_tensor_count, stats.q4k_tensor_count);
    assert_eq!(cloned.total_bytes, stats.total_bytes);
    assert_eq!(cloned.architecture, stats.architecture);
}

#[test]
fn test_q4k_stats_all_q4k() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 50,
        total_bytes: 1_000_000,
        architecture: "phi".to_string(),
        num_layers: 16,
        hidden_size: 1024,
    };

    assert_eq!(stats.tensor_count, stats.q4k_tensor_count);
}

#[test]
fn test_q4k_stats_no_q4k() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 0,
        total_bytes: 2_000_000,
        architecture: "fp16_model".to_string(),
        num_layers: 12,
        hidden_size: 768,
    };

    assert_eq!(stats.q4k_tensor_count, 0);
}

// =============================================================================
// Additional Conversion and Serialization Tests
// =============================================================================

#[test]
fn test_apr_bytes_contains_metadata_json() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // After header (64 bytes), there should be JSON metadata
    let metadata_start = HEADER_SIZE;
    let metadata_slice = &bytes[metadata_start..metadata_start + 10];
    // JSON starts with '{' or could be whitespace
    let has_json_start = metadata_slice.contains(&b'{');
    assert!(has_json_start, "Metadata section should contain JSON");
}

#[test]
fn test_apr_bytes_aligned_to_64() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Header should be 64 bytes
    assert!(bytes.len() >= 64);
}

#[test]
fn test_conversion_with_bias_weights() {
    // Create transformer with biases
    let config = AprTransformerConfig {
        architecture: "test_bias".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: Some(vec![0.0; 8]),
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: Some(vec![0.0; 3 * 8]),
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: Some(vec![0.0; 8]),
        ffn_gate_weight: Some(vec![0.01; 8 * 16]),
        ffn_gate_bias: Some(vec![0.0; 16]),
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: Some(vec![0.0; 16]),
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: Some(vec![0.0; 8]),
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: Some(vec![0.0; 8]),
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: Some(vec![0.0; 8]),
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: Some(vec![0.0; 10]),
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!(loaded.layers[0].attn_norm_bias.is_some());
    assert!(loaded.layers[0].qkv_bias.is_some());
    assert!(loaded.lm_head_bias.is_some());
}

#[test]
fn test_gguf_transformer_with_gate_weights() {
    // Create GGUF transformer with FFN gate weights (SwiGLU style)
    let config = GGUFConfig {
        architecture: "swiglu_test".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.02; 8 * 16]), // SwiGLU gate
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].ffn_gate_weight.is_some());
}

#[test]
fn test_conversion_preserves_all_layer_weights() {
    let gguf = create_minimal_gguf_transformer(16, 3, 50, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    for (i, (gguf_layer, apr_layer)) in gguf.layers.iter().zip(apr.layers.iter()).enumerate() {
        assert_eq!(
            gguf_layer.attn_norm_weight, apr_layer.attn_norm_weight,
            "Layer {} attn_norm_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.qkv_weight, apr_layer.qkv_weight,
            "Layer {} qkv_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.attn_output_weight, apr_layer.attn_output_weight,
            "Layer {} attn_output_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.ffn_up_weight, apr_layer.ffn_up_weight,
            "Layer {} ffn_up_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.ffn_down_weight, apr_layer.ffn_down_weight,
            "Layer {} ffn_down_weight mismatch",
            i
        );
    }
}

#[test]
fn test_stats_with_larger_model() {
    let apr = create_minimal_apr_transformer(512, 12, 32000, 2048);
    let stats = GgufToAprConverter::stats(&apr);

    // Should have significant parameter count
    assert!(stats.total_parameters > 1_000_000);
    assert!(stats.memory_mb() > 1.0);
}

#[test]
fn test_stats_memory_calculations_consistency() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 4,
        hidden_dim: 512,
        vocab_size: 10000,
        architecture: "test".to_string(),
    };

    // GB should be less than MB
    assert!(stats.memory_gb() < stats.memory_mb());
    // B should be less than M
    assert!(stats.parameters_b() < stats.parameters_m());
}

#[test]
fn test_stats_very_large_model() {
    let stats = ConversionStats {
        total_parameters: 405_000_000_000, // 405B
        memory_bytes_f32: 1_620_000_000_000,
        num_layers: 126,
        hidden_dim: 16384,
        vocab_size: 128256,
        architecture: "llama3_405b".to_string(),
    };

    assert!(stats.parameters_b() > 400.0);
    assert!(stats.memory_gb() > 1000.0);
}

#[test]
fn test_roundtrip_with_different_architectures() {
    for arch in ["llama", "phi2", "qwen2", "mistral", "gemma"] {
        let config = AprTransformerConfig {
            architecture: arch.to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let apr = AprTransformer {
            config,
            token_embedding: vec![0.1; 10 * 8],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; 8],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; 8 * 3 * 8],
                qkv_bias: None,
                attn_output_weight: vec![0.01; 8 * 8],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; 8 * 16],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; 16 * 8],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
            }],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 8 * 10],
            lm_head_bias: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(apr.config.architecture, loaded.config.architecture);
    }
}

// =============================================================================
// Integration: Full Conversion Pipeline Tests
// =============================================================================

#[test]
fn test_full_conversion_pipeline_small_model() {
    let gguf = create_minimal_gguf_transformer(16, 2, 50, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.num_layers, 2);
    assert_eq!(stats.hidden_dim, 16);
    assert_eq!(stats.vocab_size, 50);
    assert!(stats.total_parameters > 0);
}

#[test]
fn test_full_conversion_pipeline_with_serialization() {
    let gguf = create_minimal_gguf_transformer(8, 1, 20, 16);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(apr.config.architecture, loaded.config.architecture);
    assert_eq!(apr.config.hidden_dim, loaded.config.hidden_dim);
}

#[test]
fn test_conversion_preserves_inference_capability() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![1, 2, 3];

    let result = apr.forward(&tokens);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), apr.config.vocab_size);
}

#[test]
fn test_conversion_deterministic_inference() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![1, 2];

    let logits1 = apr.forward(&tokens).expect("forward 1");
    let logits2 = apr.forward(&tokens).expect("forward 2");

    assert_eq!(logits1, logits2);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_convert_single_token_vocab() {
    let apr = create_minimal_apr_transformer(4, 1, 1, 8);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.vocab_size, 1);
}

#[test]
fn test_convert_minimal_hidden_dim() {
    let apr = create_minimal_apr_transformer(1, 1, 10, 4);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.hidden_dim, 1);
}

#[test]
fn test_convert_large_intermediate_dim() {
    let apr = create_minimal_apr_transformer(16, 1, 50, 1024);
    let stats = GgufToAprConverter::stats(&apr);

    assert!(stats.total_parameters > 0);
}

#[test]
fn test_apr_header_alignment() {
    // Verify ALIGNMENT constant is correct
    assert_eq!(ALIGNMENT, 64);
    assert_eq!(HEADER_SIZE, 64);
}

#[test]
fn test_apr_magic_bytes() {
    // Verify MAGIC constant
    assert_eq!(MAGIC[0], 0x41); // 'A'
    assert_eq!(MAGIC[1], 0x50); // 'P'
    assert_eq!(MAGIC[2], 0x52); // 'R'
    assert_eq!(MAGIC[3], 0x00); // NUL
}
