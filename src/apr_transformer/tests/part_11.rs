//! T-COV-95 Phase 54: Q4K Fused Kernel Dispatch & Force-F32 Paths
//!
//! Covers uncovered Q4K/Q6K fused kernel dispatch in forward_with_cache and
//! forward (batch), plus APR_FORCE_F32 paths and from_apr_bytes error edges.

use crate::apr_transformer::config::Q4KLayerWeights;
use crate::apr_transformer::{
    AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer,
};

// Q4K: each row = ceil(in_dim/256) blocks × 144 bytes/block, total = out_dim rows
fn q4k_bytes(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let blocks_per_row = (in_dim + 255) / 256;
    vec![0u8; out_dim * blocks_per_row * 144]
}

// Q6K: each row = ceil(in_dim/256) blocks × 210 bytes/block, total = out_dim rows
fn q6k_bytes(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let blocks_per_row = (in_dim + 255) / 256;
    vec![0u8; out_dim * blocks_per_row * 210]
}

/// Build a minimal AprTransformer with populated q4k_layers for fused kernel paths.
/// Uses SwiGLU (has gate weight, ffn_norm) with separate Q/K/V Q4K weights.
fn build_apr_with_q4k_fused(
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
) -> AprTransformer {
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    // F32 layer weights (fallback path, also used for norm/bias)
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        // QKV as flat F32 (used when Q4K separate weights not found)
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
    };

    // Q4K layer with separate weights for fused kernel dispatch
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// Build AprTransformer with Q6K variant weights (V=Q6K, up=Q6K, down=Q6K).
fn build_apr_with_q6k_variants(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
    };

    // Q4K layer with Q6K fallback variants
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: None, // No Q4K V — forces Q6K fallback
        attn_v_weight_q6k: Some(q6k_bytes(kv_size, hidden)),
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_up_weight: None,   // No Q4K up — forces Q6K fallback
        ffn_down_weight: None, // No Q4K down — forces Q6K fallback
        ffn_down_weight_q6k: Some(q6k_bytes(hidden, intermediate)),
        ffn_up_weight_q6k: Some(q6k_bytes(intermediate, hidden)),
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// forward_with_cache: Q4K fused kernel paths
// ============================================================================

#[test]
fn test_fwc_q4k_fused_first_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K fused first token: {}",
        result.unwrap_err()
    );
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_fwc_q4k_fused_multi_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(1, &mut cache, 0).unwrap();
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q4K fused second token: {}",
        result.unwrap_err()
    );
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_fwc_q4k_fused_three_tokens() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(0, &mut cache, 0).unwrap();
    let _ = apr.forward_with_cache(1, &mut cache, 1).unwrap();
    let result = apr.forward_with_cache(2, &mut cache, 2);
    assert!(
        result.is_ok(),
        "Q4K fused third token: {}",
        result.unwrap_err()
    );
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_fwc_q4k_fused_gqa() {
    // GQA: num_kv_heads < num_heads
    let apr = build_apr_with_q4k_fused(32, 64, 4, 2, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K fused GQA first token: {}",
        result.unwrap_err()
    );

    let result2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result2.is_ok(),
        "Q4K fused GQA second token: {}",
        result2.unwrap_err()
    );
}

// ============================================================================
// forward_with_cache: Q6K variant paths (V=Q6K, up=Q6K, down=Q6K)
// ============================================================================

#[test]
fn test_fwc_q6k_v_fallback() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K V fallback: {}", result.unwrap_err());
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_fwc_q6k_variants_multi_token() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(1, &mut cache, 0).unwrap();
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q6K variants multi: {}",
        result.unwrap_err()
    );
}

// ============================================================================
// forward_with_cache: lm_head Q4K and Q6K paths
// ============================================================================

#[test]
fn test_fwc_lm_head_q4k() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Add Q4K lm_head: [vocab_size, hidden_dim]
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K lm_head: {}", result.unwrap_err());
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_fwc_lm_head_q6k() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Q6K lm_head (no Q4K lm_head)
    apr.lm_head_weight_q6k = Some(q6k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K lm_head: {}", result.unwrap_err());
    assert_eq!(result.unwrap().len(), 16);
}

// ============================================================================
// forward (batch): Q4K fused kernel paths
// ============================================================================

#[test]
fn test_forward_batch_q4k_fused_single_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K batch single: {}", result.unwrap_err());
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_forward_batch_q4k_fused_multi_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Q4K batch multi: {}", result.unwrap_err());
    let logits = result.unwrap();
    // forward() returns vocab_size logits (last token position)
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_batch_q6k_variants() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "Q6K batch: {}", result.unwrap_err());
}

#[test]
fn test_forward_batch_q4k_lm_head() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K lm_head batch: {}", result.unwrap_err());
}

// ============================================================================
// forward_with_cache: no q4k_layers (F32 fallback for all)
// ============================================================================

#[test]
fn test_fwc_f32_fallback_explicit() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.q4k_layers = None; // Force F32 fallback
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "F32 fallback: {}", result.unwrap_err());
}

// ============================================================================
// from_apr_bytes: Error path — no lm_head AND no matching embed for tying
// ============================================================================

// APR v2 constants
const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"
const HEADER_SIZE: usize = 64;

struct TensorDef {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    data: Vec<u8>,
}

/// Build APR v2 binary data using BINARY tensor index format (matching from_apr_bytes parser).
fn build_apr_v2(metadata_json: &str, tensors: &[TensorDef]) -> Vec<u8> {
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Build tensor index entries in BINARY format
    let mut index_bytes = Vec::new();
    let mut current_offset = 0u64;

    for t in tensors {
        // name_len (2) + name + dtype (1) + ndim (1) + dims (8 each) + offset (8) + size (8)
        index_bytes.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(t.name.as_bytes());
        index_bytes.push(t.dtype);
        index_bytes.push(t.dims.len() as u8);
        for &dim in &t.dims {
            index_bytes.extend_from_slice(&dim.to_le_bytes());
        }
        index_bytes.extend_from_slice(&current_offset.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
        current_offset += t.data.len() as u64;
    }

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + index_bytes.len() as u64;
    let total_data_size: usize = tensors.iter().map(|t| t.data.len()).sum();
    let total_size = data_offset as usize + total_data_size;

    let mut buf = vec![0u8; total_size];

    // Write header
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = 2; // version major
    buf[5] = 0;
    buf[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
    buf[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    buf[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    buf[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    buf[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    buf[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    buf[idx_start..idx_start + index_bytes.len()].copy_from_slice(&index_bytes);

    // Write tensor data
    let mut pos = data_offset as usize;
    for t in tensors {
        buf[pos..pos + t.data.len()].copy_from_slice(&t.data);
        pos += t.data.len();
    }

    buf
}

fn make_f32_data(n: usize, val: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

#[test]
fn test_from_apr_bytes_no_lm_head_no_embed_for_tying_fatal() {
    // Use "tok_embeddings.weight" which IS found for embed lookup but NOT tried
    // for weight tying (tying only tries model.embed_tokens.weight & token_embd.weight).
    // No lm_head.weight, no output.weight => hits line 896 error path.
    let hidden = 8;
    let intermediate = 32;
    let vocab = 4;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let kv_dim = 4 * (hidden / 4);
    let tensors = vec![
        // Embed: "tok_embeddings.weight" - found for embed, NOT for weight tying
        TensorDef {
            name: "tok_embeddings.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Norm
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // Layer norms
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // Must provide QKV + attn_output + FFN for layer construction to succeed
        TensorDef {
            name: "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            dtype: 0,
            dims: vec![(hidden + 2 * kv_dim) as u64, hidden as u64],
            data: make_f32_data((hidden + 2 * kv_dim) * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.001),
        },
    ];

    let data = build_apr_v2(&meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_err(),
        "Expected error for no lm_head + no embed for tying"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("FATAL") || err_msg.contains("lm_head"),
        "Error should mention FATAL or lm_head: {err_msg}"
    );
}

// ============================================================================
// from_apr_bytes: Q4K separate tensors → q4k_layers populated
// ============================================================================

#[test]
fn test_from_apr_bytes_separate_q4k_tensors_populates_q4k_layers() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, heads, kv_heads, vocab, intermediate
    );

    let mut tensors = vec![
        // Embedding (F32)
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Output norm (F32)
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // LM head (F32)
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Attn norm (F32)
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // FFN norm (F32)
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];

    // Add SEPARATE Q4K attention weights
    // Q4K tensors: (name, out_dim, in_dim) — data size = out_dim * ceil(in_dim/256) * 144
    let q4k_tensors: Vec<(&str, usize, usize)> = vec![
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.v_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ];

    for (name, out_dim, in_dim) in q4k_tensors {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12, // Q4_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Separate Q4K parse failed");

    // Verify q4k_layers is populated
    assert!(apr.q4k_layers.is_some(), "q4k_layers should be Some");
    let q4k = &apr.q4k_layers.as_ref().unwrap()[0];
    assert!(
        q4k.attn_q_weight.is_some(),
        "attn_q_weight should be populated"
    );
    assert!(
        q4k.attn_k_weight.is_some(),
        "attn_k_weight should be populated"
    );
    assert!(
        q4k.attn_v_weight.is_some(),
        "attn_v_weight should be populated"
    );
    assert!(
        q4k.attn_output_weight.is_some(),
        "attn_output_weight should be populated"
    );
    assert!(
        q4k.ffn_gate_weight.is_some(),
        "ffn_gate_weight should be populated"
    );
    assert!(
        q4k.ffn_up_weight.is_some(),
        "ffn_up_weight should be populated"
    );
    assert!(
        q4k.ffn_down_weight.is_some(),
        "ffn_down_weight should be populated"
    );
}

#[test]
fn test_from_apr_bytes_separate_q4k_then_forward_with_cache() {
    // Build APR with separate Q4K tensors, then run forward_with_cache
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, heads, kv_heads, vocab, intermediate
    );

    let mut tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];

    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.v_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12,
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors)).expect("Parse failed");
    assert!(apr.q4k_layers.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Separate Q4K fwc: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Separate Q4K fwc 2nd: {}", r2.unwrap_err());
}

// ============================================================================
// from_apr_bytes: Q6K separate tensors for mixed quant
// ============================================================================

#[test]
fn test_from_apr_bytes_mixed_q4k_q6k_tensors() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let mut tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];

    // Q4K for Q, K, attn_output, gate
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12, // Q4_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    // Q6K for V, up, down (mixed quantization)
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.v_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 14, // Q6_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q6k_bytes(out_dim, in_dim),
        });
    }

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors))
        .expect("Mixed Q4K/Q6K parse failed");

    assert!(apr.q4k_layers.is_some());
    let q4k = &apr.q4k_layers.as_ref().unwrap()[0];
    assert!(q4k.attn_q_weight.is_some());
    assert!(q4k.attn_v_weight.is_none(), "V should NOT be Q4K");
    assert!(q4k.attn_v_weight_q6k.is_some(), "V should be Q6K");
    assert!(q4k.ffn_up_weight.is_none(), "up should NOT be Q4K");
    assert!(q4k.ffn_up_weight_q6k.is_some(), "up should be Q6K");
    assert!(q4k.ffn_down_weight.is_none(), "down should NOT be Q4K");
    assert!(q4k.ffn_down_weight_q6k.is_some(), "down should be Q6K");
}

// ============================================================================
// generate: Q4K model
// ============================================================================

#[test]
fn test_generate_q4k_fused() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.generate(&[1, 2], 3);
    assert!(result.is_ok(), "Q4K generate: {}", result.unwrap_err());
    let tokens = result.unwrap();
    assert!(tokens.len() >= 2, "Should have at least input tokens");
}

#[test]
fn test_generate_q6k_variants() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let result = apr.generate(&[1], 2);
    assert!(result.is_ok(), "Q6K generate: {}", result.unwrap_err());
}

// ============================================================================
// forward_with_cache: GELU model with Q4K fused (no gate weight)
// ============================================================================

#[test]
fn test_fwc_q4k_gelu_no_gate() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]), // LayerNorm (has bias)
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: Some(vec![0.0; hidden + 2 * kv_size]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU: no gate
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
    };

    // Q4K layers for GELU model (no gate)
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None, // No gate for GELU
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K GELU fwc: {}", result.unwrap_err());
}

// ============================================================================
// forward (batch): GELU model with Q4K
// ============================================================================

#[test]
fn test_forward_batch_q4k_gelu() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
    };

    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None, // No separate — falls through to combined QKV
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None,
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "Q4K GELU batch: {}", result.unwrap_err());
}

// ============================================================================
// Env-var-gated paths: REALIZE_TRACE, APR_FORCE_F32, REALIZE_DEBUG
// These tests cover trace/debug/force_f32 conditional blocks that are
// otherwise unreachable without setting environment variables.
// ============================================================================

#[test]
fn test_fwc_with_realize_trace() {
    // REALIZE_TRACE enables eprintln trace blocks in forward_with_cache
    // Safe in parallel: only affects stderr output, not computation
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Trace fwc first: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Trace fwc second: {}", r2.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_fwc_force_f32_with_q4k_layers() {
    // APR_FORCE_F32 forces F32 fallback even when Q4K layers exist
    unsafe {
        std::env::set_var("APR_FORCE_F32", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Force F32 fwc: {}", result.unwrap_err());
    unsafe {
        std::env::remove_var("APR_FORCE_F32");
    }
}

#[test]
fn test_fwc_force_f32_with_trace() {
    // Both APR_FORCE_F32 and REALIZE_TRACE — covers the force_f32 trace blocks
    unsafe {
        std::env::set_var("APR_FORCE_F32", "1");
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Force F32 + trace: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Force F32 + trace 2nd: {}", r2.unwrap_err());
    unsafe {
        std::env::remove_var("APR_FORCE_F32");
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_forward_batch_with_realize_trace() {
    // REALIZE_TRACE for batch forward path
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Trace batch: {}", result.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_from_apr_bytes_with_realize_debug() {
    // REALIZE_DEBUG enables debug eprintln blocks in from_apr_bytes
    unsafe {
        std::env::set_var("REALIZE_DEBUG", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let head_dim = hidden / 4;
    let kv_size = 4 * head_dim;
    let mut tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];
    // Add Q4K separate weights to trigger Q4K debug path
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.v_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12,
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors));
    assert!(apr.is_ok(), "Debug from_apr_bytes: {}", apr.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_DEBUG");
    }
}

// ============================================================================
// from_apr_bytes: GGUF naming (output.weight, blk.X) triggers is_gguf_model
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_naming() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let head_dim = hidden / 4;
    let kv_dim = 4 * head_dim; // kv_heads=4

    let tensors = vec![
        // GGUF-style embedding name
        TensorDef {
            name: "token_embd.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // GGUF-style output norm
        TensorDef {
            name: "output_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // GGUF-style lm_head
        TensorDef {
            name: "output.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // GGUF-style layer norms
        TensorDef {
            name: "blk.0.attn_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // GGUF-style QKV (combined)
        TensorDef {
            name: "blk.0.attn_qkv.weight".to_string(),
            dtype: 0,
            dims: vec![(hidden + 2 * kv_dim) as u64, hidden as u64],
            data: make_f32_data((hidden + 2 * kv_dim) * hidden, 0.001),
        },
        // GGUF-style attn output
        TensorDef {
            name: "blk.0.attn_output.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.001),
        },
        // GGUF-style FFN (SwiGLU)
        TensorDef {
            name: "blk.0.ffn_gate.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.001),
        },
    ];

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors));
    assert!(apr.is_ok(), "GGUF naming: {}", apr.unwrap_err());
    let model = apr.unwrap();
    assert_eq!(model.config.hidden_dim, hidden);
    assert_eq!(model.config.vocab_size, vocab);
    assert_eq!(model.layers.len(), 1);
}

// ============================================================================
// forward (batch): GELU F32-only (no Q4K) — covers ffn_up_bias/ffn_down_bias
// ============================================================================

#[test]
fn test_forward_batch_gelu_f32_no_q4k() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: None, // Pure F32 — exercises F32 GELU FFN with bias
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "GELU F32 batch: {}", result.unwrap_err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), vocab);
}

// ============================================================================
// forward_with_cache: Q6K lm_head (different from Q4K lm_head path)
// ============================================================================

#[test]
fn test_fwc_q6k_lm_head_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q6k = Some(q6k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q6K lm_head + trace: {}",
        result.unwrap_err()
    );
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: Q4K lm_head with trace
// ============================================================================

#[test]
fn test_fwc_q4k_lm_head_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K lm_head + trace: {}",
        result.unwrap_err()
    );
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: Q6K variants with trace (covers Q6K trace blocks)
// ============================================================================

#[test]
fn test_fwc_q6k_variants_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Q6K variants + trace: {}", r1.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: partial Q4K (some fields None) + trace
// Covers the F32 fallback trace blocks INSIDE the `if !force_f32` branch
// (e.g., "attn_output using F32 fallback (slow!)")
// ============================================================================

#[test]
fn test_fwc_partial_q4k_f32_fallback_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
    };

    // Partial Q4K: only Q and K populated, everything else None
    // This forces F32 fallback for V, attn_output, gate, up, down
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: None,      // F32 fallback
        attn_v_weight_q6k: None,  // no Q6K either
        attn_output_weight: None, // F32 fallback
        ffn_gate_weight: None,    // F32 fallback
        ffn_up_weight: None,      // F32 fallback
        ffn_down_weight: None,    // F32 fallback
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Partial Q4K + trace: {}", r1.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): out-of-vocab token + debug traces
// ============================================================================

#[test]
fn test_forward_batch_oov_token_with_debug() {
    unsafe {
        std::env::set_var("REALIZE_DEBUG", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Token 999 is way out of vocab (vocab=16)
    let result = apr.forward(&[999]);
    assert!(result.is_ok(), "OOV batch: {}", result.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_DEBUG");
    }
}

// ============================================================================
// forward_with_cache: GELU model (no gate) F32-only + trace
// Covers GELU branches in forward_with_cache with trace enabled
// ============================================================================

#[test]
fn test_fwc_gelu_f32_no_q4k_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "GELU F32 + trace: {}", result.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): GELU + trace (covers batch GELU trace paths)
// ============================================================================

#[test]
fn test_forward_batch_gelu_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(
        result.is_ok(),
        "GELU batch + trace: {}",
        result.unwrap_err()
    );
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): partial Q4K + trace (batch forward F32 fallback trace)
// ============================================================================

#[test]
fn test_forward_batch_partial_q4k_with_trace() {
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
    };

    // Only attn_output Q4K populated — all others fall through to F32
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None,
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None,
        ffn_up_weight: None,
        ffn_down_weight: None,
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2, 3]);
    assert!(
        result.is_ok(),
        "Partial Q4K batch + trace: {}",
        result.unwrap_err()
    );
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}
