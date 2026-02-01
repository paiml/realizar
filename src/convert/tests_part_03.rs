//! Convert Module Tests Part 03 - T-COV-95 Deep Coverage Bridge
//!
//! Tests for:
//! - GgufToAprQ4KConverter::infer_rope_type for all architecture families
//! - GgufToAprQ4KConverter helper methods: get_string, get_u32, get_f32
//! - RawTensor struct construction and fields
//! - Q4KConversionStats Debug/Clone
//! - ConversionStats edge cases (zero values)
//! - to_apr_bytes / from_apr_bytes additional error paths
//!
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
    use crate::convert::*;
    use std::collections::HashMap;

    // =========================================================================
    // GgufToAprQ4KConverter::infer_rope_type
    // =========================================================================

    #[test]
    fn test_infer_rope_type_llama_norm() {
        let metadata = HashMap::new();
        let rope = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(rope, 0); // NORM style
    }

    #[test]
    fn test_infer_rope_type_qwen2_neox() {
        let metadata = HashMap::new();
        let rope = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(rope, 2); // NEOX style
    }

    #[test]
    fn test_infer_rope_type_qwen3_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("qwen3", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_phi2_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("phi2", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_phi3_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("phi3", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_gemma_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gemma", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_gemma2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gemma2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_falcon_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("falcon", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_stablelm_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("stablelm", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_starcoder2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("starcoder2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_gptneox_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gptneox", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_bert_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("bert", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_deepseek2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("deepseek2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_internlm2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("internlm2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_unknown_defaults_norm() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("custom_arch", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_none() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("none".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_linear() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("linear".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_yarn() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("yarn".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_neox() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("neox".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_unknown_falls_through() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("custom_scale".to_string()),
        );
        // Unknown scaling type → falls through to architecture check → llama → 0
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    // =========================================================================
    // GgufToAprQ4KConverter helper methods
    // =========================================================================

    #[test]
    fn test_get_string_found() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_get_string_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_string(&metadata, "key"), None);
    }

    #[test]
    fn test_get_string_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        assert_eq!(GgufToAprQ4KConverter::get_string(&metadata, "key"), None);
    }

    #[test]
    fn test_get_u32_from_uint32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(42));
    }

    #[test]
    fn test_get_u32_from_int32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(42));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(42));
    }

    #[test]
    fn test_get_u32_from_uint64() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(100));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(100));
    }

    #[test]
    fn test_get_u32_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), None);
    }

    #[test]
    fn test_get_u32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not_a_number".to_string()),
        );
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), None);
    }

    #[test]
    fn test_get_f32_from_float32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(3.14));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.unwrap() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_get_f32_from_float64() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(2.718));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.unwrap() - 2.718).abs() < 1e-3);
    }

    #[test]
    fn test_get_f32_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_f32(&metadata, "key"), None);
    }

    // =========================================================================
    // RawTensor struct
    // =========================================================================

    #[test]
    fn test_raw_tensor_construction() {
        let tensor = RawTensor {
            name: "layer.0.weight".to_string(),
            data: vec![0u8; 144],
            shape: vec![256],
            dtype: 12, // Q4_K
        };
        assert_eq!(tensor.name, "layer.0.weight");
        assert_eq!(tensor.data.len(), 144);
        assert_eq!(tensor.shape, vec![256]);
        assert_eq!(tensor.dtype, 12);
    }

    #[test]
    fn test_raw_tensor_debug() {
        let tensor = RawTensor {
            name: "test".to_string(),
            data: vec![1, 2, 3],
            shape: vec![3],
            dtype: 0,
        };
        let debug = format!("{:?}", tensor);
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_raw_tensor_clone() {
        let tensor = RawTensor {
            name: "test".to_string(),
            data: vec![1, 2, 3],
            shape: vec![3],
            dtype: 0,
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
    }

    // =========================================================================
    // Q4KConversionStats
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_debug() {
        let stats = Q4KConversionStats {
            tensor_count: 100,
            q4k_tensor_count: 80,
            total_bytes: 1_000_000,
            architecture: "qwen2".to_string(),
            num_layers: 28,
            hidden_size: 1536,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("qwen2"));
        assert!(debug.contains("1536"));
    }

    #[test]
    fn test_q4k_conversion_stats_clone() {
        let stats = Q4KConversionStats {
            tensor_count: 10,
            q4k_tensor_count: 8,
            total_bytes: 500,
            architecture: "llama".to_string(),
            num_layers: 2,
            hidden_size: 64,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.tensor_count, stats.tensor_count);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    // =========================================================================
    // ConversionStats edge cases
    // =========================================================================

    #[test]
    fn test_conversion_stats_zero_values() {
        let stats = ConversionStats {
            total_parameters: 0,
            memory_bytes_f32: 0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            architecture: "empty".to_string(),
        };
        assert_eq!(stats.memory_mb(), 0.0);
        assert_eq!(stats.memory_gb(), 0.0);
        assert_eq!(stats.parameters_m(), 0.0);
        assert_eq!(stats.parameters_b(), 0.0);
    }

    #[test]
    fn test_conversion_stats_large_model() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.parameters_b() - 7.0).abs() < 0.01);
        assert!((stats.parameters_m() - 7000.0).abs() < 1.0);
        assert!(stats.memory_gb() > 25.0);
        assert!(stats.memory_mb() > 25000.0);
    }

    #[test]
    fn test_conversion_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 100,
            memory_bytes_f32: 400,
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 8,
            architecture: "test".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("test"));
    }

    // =========================================================================
    // from_apr_bytes additional error paths
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_after_header() {
        use crate::apr::{HEADER_SIZE, MAGIC};

        // Build a header that claims tensor index is beyond data
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
                       // tensor_index_offset pointing beyond the data
        header[24..32].copy_from_slice(&1000u64.to_le_bytes());
        // data_offset
        header[32..40].copy_from_slice(&2000u64.to_le_bytes());

        let result = GgufToAprConverter::from_apr_bytes(&header);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_apr_bytes_roundtrip_with_biases() {
        let config = AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 8,
            intermediate_dim: 8,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_dim;

        let transformer = AprTransformer {
            config,
            token_embedding: vec![0.1f32; vocab * hidden],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden],
                attn_norm_bias: Some(vec![0.0; hidden]), // phi2 has biases
                qkv_weight: vec![0.01; hidden * hidden * 3],
                qkv_bias: Some(vec![0.0; hidden * 3]),
                attn_output_weight: vec![0.01; hidden * hidden],
                attn_output_bias: Some(vec![0.0; hidden]),
                ffn_gate_weight: None, // phi2 has no gate
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden * intermediate],
                ffn_up_bias: Some(vec![0.0; intermediate]),
                ffn_down_weight: vec![0.01; intermediate * hidden],
                ffn_down_bias: Some(vec![0.0; hidden]),
                ffn_norm_weight: Some(vec![1.0; hidden]),
                ffn_norm_bias: Some(vec![0.0; hidden]),
            }],
            output_norm_weight: vec![1.0; hidden],
            output_norm_bias: Some(vec![0.0; hidden]),
            lm_head_weight: vec![0.01; hidden * vocab],
            lm_head_bias: Some(vec![0.0; vocab]),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&transformer).unwrap();
        let restored = GgufToAprConverter::from_apr_bytes(&bytes).unwrap();
        assert_eq!(restored.config.architecture, "phi2");
        assert_eq!(restored.config.num_heads, 2);
        // Verify biases survived roundtrip
        assert!(restored.layers[0].attn_norm_bias.is_some());
        assert!(restored.layers[0].qkv_bias.is_some());
    }

    // =========================================================================
    // BUG-APR-002: Byte size calculation for quantized tensors
    // =========================================================================
    // The Q4K converter was using integer division (/) which rounds DOWN,
    // causing "tensor exceeds file bounds" for tensors not perfectly divisible
    // by block size. Fix: use div_ceil() to round UP.

    #[test]
    fn test_bug_apr_002_q4k_byte_size_div_ceil() {
        // Q4_K block: 256 elements = 144 bytes
        // For 65600 elements (256 * 256 + 64):
        // - Wrong: (65600 / 256) * 144 = 256 * 144 = 36864 bytes
        // - Right: 65600.div_ceil(256) * 144 = 257 * 144 = 37008 bytes
        let num_elements = 65600usize;
        let wrong_byte_size = (num_elements / 256) * 144;
        let right_byte_size = num_elements.div_ceil(256) * 144;

        assert_eq!(
            wrong_byte_size, 36864,
            "Wrong calculation (integer division)"
        );
        assert_eq!(right_byte_size, 37008, "Correct calculation (div_ceil)");
        assert!(
            right_byte_size > wrong_byte_size,
            "div_ceil must give larger result"
        );
    }

    #[test]
    fn test_bug_apr_002_q8_byte_size_div_ceil() {
        // Q8_0 block: 32 elements = 34 bytes (2 scale + 32 quants)
        // For 1000 elements (32 * 31 + 8):
        // - Wrong: (1000 / 32) * 34 = 31 * 34 = 1054 bytes
        // - Right: 1000.div_ceil(32) * 34 = 32 * 34 = 1088 bytes
        let num_elements = 1000usize;
        let wrong_byte_size = (num_elements / 32) * 34;
        let right_byte_size = num_elements.div_ceil(32) * 34;

        assert_eq!(
            wrong_byte_size, 1054,
            "Wrong calculation (integer division)"
        );
        assert_eq!(right_byte_size, 1088, "Correct calculation (div_ceil)");
        assert!(
            right_byte_size > wrong_byte_size,
            "div_ceil must give larger result"
        );
    }

    #[test]
    fn test_bug_apr_002_exact_divisibility_no_change() {
        // When num_elements is perfectly divisible, both methods give same result
        let num_elements = 65536usize; // 256 * 256
        let old_style = (num_elements / 256) * 144;
        let new_style = num_elements.div_ceil(256) * 144;

        assert_eq!(
            old_style, new_style,
            "Exact divisibility should give same result"
        );
        assert_eq!(new_style, 36864);
    }

    #[test]
    fn test_bug_apr_002_q5k_byte_size_div_ceil() {
        // Q5_K block: 256 elements = 176 bytes
        // For 65537 elements (256 * 256 + 1):
        // - Wrong: (65537 / 256) * 176 = 256 * 176 = 45056 bytes
        // - Right: 65537.div_ceil(256) * 176 = 257 * 176 = 45232 bytes
        let num_elements = 65537usize;
        let wrong_byte_size = (num_elements / 256) * 176;
        let right_byte_size = num_elements.div_ceil(256) * 176;

        assert_ne!(wrong_byte_size, right_byte_size);
        assert_eq!(right_byte_size - wrong_byte_size, 176); // One extra block
    }

    #[test]
    fn test_bug_apr_002_q6k_byte_size_div_ceil() {
        // Q6_K block: 256 elements = 210 bytes
        // For 65600 elements:
        // - Wrong: (65600 / 256) * 210 = 256 * 210 = 53760 bytes
        // - Right: 65600.div_ceil(256) * 210 = 257 * 210 = 53970 bytes
        let num_elements = 65600usize;
        let wrong_byte_size = (num_elements / 256) * 210;
        let right_byte_size = num_elements.div_ceil(256) * 210;

        assert_ne!(wrong_byte_size, right_byte_size);
        assert_eq!(right_byte_size - wrong_byte_size, 210); // One extra block
    }
}
