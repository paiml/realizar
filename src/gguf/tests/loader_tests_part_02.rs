//! GGUF Loader Tests Part 02: Tensor Loading, Metadata Extraction, Model Validation
//!
//! This module provides comprehensive test coverage for the second half of loader.rs:
//! - `GGUFTransformer::from_gguf` and `load_layer`
//! - `OwnedQuantizedModel::new_for_test`
//! - `GGUFConfig::from_gguf` edge cases
//! - Transformer layer loading (both fused and separate QKV)
//! - Model validation and error handling
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::test_factory::{
    build_minimal_llama_gguf, build_minimal_phi2_gguf, create_f32_embedding_data,
    create_f32_norm_weights, create_q4_k_data, GGUFBuilder,
};
use crate::gguf::{
    GGUFConfig, GGUFModel, GGUFTransformer, GGUFValue, OwnedQKVWeights, OwnedQuantizedLayer,
    OwnedQuantizedModel, OwnedQuantizedTensor, GGUF_TYPE_F32, GGUF_TYPE_Q4_K,
};

// =============================================================================
// GGUFTransformer Loading Tests
// =============================================================================

#[test]
fn test_loader_part02_transformer_from_minimal_llama() {
    // Build a minimal valid LLaMA-style GGUF model
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);

    let model = GGUFModel::from_bytes(&data).expect("Should parse GGUF");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    assert_eq!(transformer.config.architecture, "llama");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);
    assert_eq!(transformer.config.num_kv_heads, 4);
    assert!(!transformer.token_embedding.is_empty());
    assert_eq!(transformer.layers.len(), 1);
    assert!(!transformer.output_norm_weight.is_empty());
    assert!(!transformer.lm_head_weight.is_empty());
}

#[test]
fn test_loader_part02_transformer_from_minimal_phi2() {
    // Build a minimal valid Phi-2 style GGUF model (fused QKV)
    let data = build_minimal_phi2_gguf(100, 64, 128, 4);

    let model = GGUFModel::from_bytes(&data).expect("Should parse GGUF");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    assert_eq!(transformer.config.architecture, "phi2");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.layers.len(), 1);
    // Phi-2 uses fused QKV, so qkv_weight should be a single large tensor
    let layer = &transformer.layers[0];
    // Layer has concatenated QKV weights
    assert!(!layer.qkv_weight.is_empty());
}

#[test]
fn test_loader_part02_transformer_missing_embedding() {
    // Build GGUF without token embedding
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 0)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse GGUF");
    let result = GGUFTransformer::from_gguf(&model, &data);

    assert!(result.is_err(), "Missing embedding should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("token_embd") || err.contains("not found"),
        "Error should mention token_embd: {}",
        err
    );
}

#[test]
fn test_loader_part02_transformer_layer_with_bias() {
    // Create model with bias tensors
    let vocab_size = 100;
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let kv_dim = hidden_dim; // MHA

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let bias_data = vec![0.1f32; hidden_dim];

    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_f32_tensor("blk.0.attn_norm.bias", &[hidden_dim as u64], &bias_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load");

    // Check that bias was loaded
    let layer = &transformer.layers[0];
    assert!(layer.attn_norm_bias.is_some());
}

#[test]
fn test_loader_part02_transformer_tied_embeddings() {
    // Build model without output.weight (tied embeddings fallback)
    let vocab_size = 100;
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let kv_dim = hidden_dim;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        // Note: no output.weight - should fall back to token_embd.weight
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load with tied embeddings");

    // lm_head_weight should be same as token_embedding (tied)
    assert_eq!(transformer.lm_head_weight.len(), transformer.token_embedding.len());
}

// =============================================================================
// GGUFConfig Tests
// =============================================================================

#[test]
fn test_loader_part02_config_from_minimal_model() {
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let config = GGUFConfig::from_gguf(&model).expect("Should extract config");

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.num_layers, 1);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.context_length, 256);
    assert!((config.rope_theta - 10000.0).abs() < 1.0);
    assert!((config.eps - 1e-5).abs() < 1e-7);
}

#[test]
fn test_loader_part02_config_default_values() {
    // Create model with minimal metadata to test defaults
    let vocab_size = 100;
    let hidden_dim = 64;
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("test")
        .hidden_dim("test", hidden_dim as u32)
        .num_layers("test", 2)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let config = GGUFConfig::from_gguf(&model).expect("Should extract config");

    // Check defaults are applied
    assert_eq!(config.architecture, "test");
    assert_eq!(config.num_heads, hidden_dim / 64); // default: hidden_dim / 64
    assert_eq!(config.context_length, 2048); // default
    assert!((config.rope_theta - 10000.0).abs() < 1.0); // default
    assert!((config.eps - 1e-5).abs() < 1e-7); // default
}

#[test]
fn test_loader_part02_config_qwen2_style() {
    let vocab_size = 100;
    let hidden_dim = 64;
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("qwen2")
        .hidden_dim("qwen2", hidden_dim as u32)
        .num_layers("qwen2", 2)
        .num_heads("qwen2", 8)
        .num_kv_heads("qwen2", 2)
        .context_length("qwen2", 32768)
        .rope_freq_base("qwen2", 1_000_000.0)
        .rms_epsilon("qwen2", 1e-6)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let config = GGUFConfig::from_gguf(&model).expect("Should extract config");

    assert_eq!(config.architecture, "qwen2");
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.context_length, 32768);
    assert!((config.rope_theta - 1_000_000.0).abs() < 100.0);
    assert!((config.eps - 1e-6).abs() < 1e-8);
    assert_eq!(config.rope_type, 2); // NEOX style for qwen2
}

// =============================================================================
// OwnedQuantizedModel Tests
// =============================================================================

#[test]
fn test_loader_part02_owned_model_new_for_test() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let token_embedding = vec![0.1f32; 50 * 32];
    let layers = vec![OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; 32],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 96,
            qtype: GGUF_TYPE_Q4_K,
        }),
        qkv_bias: None,
        attn_output_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        attn_output_bias: None,
        ffn_up_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 64,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_up_bias: None,
        ffn_down_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; 32]),
        ffn_norm_bias: None,
    }];
    let output_norm_weight = vec![1.0f32; 32];
    let lm_head_weight = OwnedQuantizedTensor {
        data: vec![0u8; 144],
        in_dim: 32,
        out_dim: 50,
        qtype: GGUF_TYPE_Q4_K,
    };

    let model = OwnedQuantizedModel::new_for_test(
        config.clone(),
        token_embedding.clone(),
        layers,
        output_norm_weight.clone(),
        None,
        lm_head_weight,
        None,
    );

    assert_eq!(model.config.architecture, "test");
    assert_eq!(model.config.hidden_dim, 32);
    assert_eq!(model.token_embedding.len(), 50 * 32);
    assert_eq!(model.layers.len(), 1);
    assert_eq!(model.output_norm_weight.len(), 32);
}

#[test]
fn test_loader_part02_owned_model_with_biases() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 2,
    };

    let token_embedding = vec![0.1f32; 50 * 32];
    let layers = vec![OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; 32],
        attn_norm_bias: Some(vec![0.0f32; 32]),
        qkv_weight: OwnedQKVWeights::Fused(OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 96,
            qtype: GGUF_TYPE_Q4_K,
        }),
        qkv_bias: Some(vec![0.0f32; 96]),
        attn_output_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        attn_output_bias: Some(vec![0.0f32; 32]),
        ffn_up_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 64,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_up_bias: Some(vec![0.0f32; 64]),
        ffn_down_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_down_bias: Some(vec![0.0f32; 32]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
    }];
    let output_norm_weight = vec![1.0f32; 32];
    let output_norm_bias = Some(vec![0.0f32; 32]);
    let lm_head_weight = OwnedQuantizedTensor {
        data: vec![0u8; 144],
        in_dim: 32,
        out_dim: 50,
        qtype: GGUF_TYPE_Q4_K,
    };
    let lm_head_bias = Some(vec![0.0f32; 50]);

    let model = OwnedQuantizedModel::new_for_test(
        config,
        token_embedding,
        layers,
        output_norm_weight,
        output_norm_bias,
        lm_head_weight,
        lm_head_bias,
    );

    assert!(model.output_norm_bias.is_some());
    assert!(model.lm_head_bias.is_some());
    assert!(model.layers[0].attn_norm_bias.is_some());
    assert!(model.layers[0].qkv_bias.is_some());
}

#[test]
fn test_loader_part02_owned_qkv_separate() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let kv_dim = 16; // num_kv_heads * head_dim

    let layers = vec![OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; 32],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: OwnedQuantizedTensor {
                data: vec![0u8; 144],
                in_dim: 32,
                out_dim: 32,
                qtype: GGUF_TYPE_Q4_K,
            },
            k: OwnedQuantizedTensor {
                data: vec![0u8; 144],
                in_dim: 32,
                out_dim: kv_dim,
                qtype: GGUF_TYPE_Q4_K,
            },
            v: OwnedQuantizedTensor {
                data: vec![0u8; 144],
                in_dim: 32,
                out_dim: kv_dim,
                qtype: GGUF_TYPE_Q4_K,
            },
        },
        qkv_bias: None,
        attn_output_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        attn_output_bias: None,
        ffn_up_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 64,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_up_bias: None,
        ffn_down_weight: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 32,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_down_bias: None,
        ffn_gate_weight: Some(OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 64,
            qtype: GGUF_TYPE_Q4_K,
        }),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; 32]),
        ffn_norm_bias: None,
    }];

    let model = OwnedQuantizedModel::new_for_test(
        config,
        vec![0.1f32; 50 * 32],
        layers,
        vec![1.0f32; 32],
        None,
        OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 32,
            out_dim: 50,
            qtype: GGUF_TYPE_Q4_K,
        },
        None,
    );

    // Check separate QKV
    match &model.layers[0].qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            assert_eq!(q.out_dim, 32);
            assert_eq!(k.out_dim, kv_dim);
            assert_eq!(v.out_dim, kv_dim);
        }
        OwnedQKVWeights::Fused(_) => panic!("Expected separate QKV"),
    }

    // Check FFN gate is present
    assert!(model.layers[0].ffn_gate_weight.is_some());
}

// =============================================================================
// OwnedQKVWeights Tests
// =============================================================================

#[test]
fn test_loader_part02_owned_qkv_fused_dims() {
    let qkv = OwnedQKVWeights::Fused(OwnedQuantizedTensor {
        data: vec![0u8; 144],
        in_dim: 64,
        out_dim: 192, // 3 * 64
        qtype: GGUF_TYPE_Q4_K,
    });

    assert_eq!(qkv.out_dim(), 192);
    assert_eq!(qkv.q_dim(), 64); // 192 / 3
}

#[test]
fn test_loader_part02_owned_qkv_separate_dims() {
    let qkv = OwnedQKVWeights::Separate {
        q: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 64,
            qtype: GGUF_TYPE_Q4_K,
        },
        k: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 16,
            qtype: GGUF_TYPE_Q4_K,
        },
        v: OwnedQuantizedTensor {
            data: vec![0u8; 144],
            in_dim: 64,
            out_dim: 16,
            qtype: GGUF_TYPE_Q4_K,
        },
    };

    assert_eq!(qkv.out_dim(), 64 + 16 + 16); // 96
    assert_eq!(qkv.q_dim(), 64);
}

// =============================================================================
// GGUFTransformerLayer Structure Tests
// =============================================================================

#[test]
fn test_loader_part02_layer_qkv_concat() {
    // Test that separate Q, K, V tensors are correctly concatenated
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load");

    let layer = &transformer.layers[0];
    // LLaMA style uses separate Q, K, V so qkv_weight is concatenated
    // The concatenated size should be q_size + k_size + v_size
    let hidden_dim = 64;
    let kv_dim = 64; // Same as hidden for this test
    let expected_qkv_size = hidden_dim * hidden_dim + hidden_dim * kv_dim + hidden_dim * kv_dim;
    assert_eq!(layer.qkv_weight.len(), expected_qkv_size);
}

#[test]
fn test_loader_part02_layer_ffn_components() {
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load");

    let layer = &transformer.layers[0];

    // LLaMA-style models have FFN gate (SwiGLU)
    assert!(layer.ffn_gate_weight.is_some());

    // FFN dimensions
    // up: hidden -> intermediate
    // down: intermediate -> hidden
    // gate: hidden -> intermediate
    let intermediate_dim = 128;
    let hidden_dim = 64;

    assert_eq!(layer.ffn_up_weight.len(), hidden_dim * intermediate_dim);
    assert_eq!(layer.ffn_down_weight.len(), intermediate_dim * hidden_dim);
    assert_eq!(
        layer.ffn_gate_weight.as_ref().unwrap().len(),
        hidden_dim * intermediate_dim
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_loader_part02_config_missing_num_layers() {
    let vocab_size = 100;
    let hidden_dim = 64;
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("test")
        .hidden_dim("test", hidden_dim as u32)
        // Missing num_layers (block_count)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = GGUFConfig::from_gguf(&model);

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("block_count") || err.contains("missing"),
        "Error: {}",
        err
    );
}

#[test]
fn test_loader_part02_config_missing_hidden_dim() {
    let data = GGUFBuilder::new()
        .architecture("test")
        .num_layers("test", 2)
        // Missing hidden_dim (embedding_length)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = GGUFConfig::from_gguf(&model);

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("embedding_length") || err.contains("missing"),
        "Error: {}",
        err
    );
}

// =============================================================================
// Multi-Layer Tests
// =============================================================================

#[test]
fn test_loader_part02_multi_layer_model() {
    // Build a 2-layer model
    let vocab_size = 50;
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let num_layers = 2;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * hidden_dim);
    let v_data = create_q4_k_data(hidden_dim * hidden_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let mut builder = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", num_layers as u32)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        );

    // Add layers
    for i in 0..num_layers {
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", i),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_q.weight", i),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_k.weight", i),
                &[hidden_dim as u64, hidden_dim as u64],
                &k_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_v.weight", i),
                &[hidden_dim as u64, hidden_dim as u64],
                &v_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", i),
                &[hidden_dim as u64, hidden_dim as u64],
                &attn_out_data,
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", i),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", i),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_up_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", i),
                &[intermediate_dim as u64, hidden_dim as u64],
                &ffn_down_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_gate.weight", i),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_gate_data,
            );
    }

    let data = builder
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("Should load");

    assert_eq!(transformer.layers.len(), num_layers);
    assert_eq!(transformer.config.num_layers, num_layers);
}

// =============================================================================
// Config Clone and Debug Tests
// =============================================================================

#[test]
fn test_loader_part02_config_clone() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 8,
        num_heads: 16,
        num_kv_heads: 4,
        vocab_size: 32000,
        intermediate_dim: 1024,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let cloned = config.clone();

    assert_eq!(cloned.architecture, config.architecture);
    assert_eq!(cloned.hidden_dim, config.hidden_dim);
    assert_eq!(cloned.num_layers, config.num_layers);
    assert_eq!(cloned.num_heads, config.num_heads);
    assert_eq!(cloned.num_kv_heads, config.num_kv_heads);
    assert_eq!(cloned.vocab_size, config.vocab_size);
    assert_eq!(cloned.intermediate_dim, config.intermediate_dim);
    assert_eq!(cloned.context_length, config.context_length);
    assert!((cloned.rope_theta - config.rope_theta).abs() < f32::EPSILON);
    assert!((cloned.eps - config.eps).abs() < f32::EPSILON);
    assert_eq!(cloned.rope_type, config.rope_type);
}

#[test]
fn test_loader_part02_config_debug() {
    let config = GGUFConfig {
        architecture: "debug_test".to_string(),
        hidden_dim: 512,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 2,
        vocab_size: 50000,
        intermediate_dim: 2048,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-6,
        rope_type: 2,
    };

    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("debug_test"));
    assert!(debug_str.contains("512"));
    assert!(debug_str.contains("GGUFConfig"));
}
