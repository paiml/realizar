//! T-COV-95 Phase 53: AprTransformer::from_apr_bytes dtype coverage
//!
//! Covers the uncovered dtype dispatch branches in the `get_f32_tensor` closure
//! within `from_apr_bytes`: Q4_K(12), Q5_K(13), Q6_K(14), Q8_0(8), F16(1).
//! Also covers weight tying, GGUF naming, metadata aliases, and Q4K raw bytes paths.

use crate::apr_transformer::AprTransformer;

// APR v2 constants
const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"
const HEADER_SIZE: usize = 64;

// GGML dtype constants
const DTYPE_F32: u8 = 0;
const DTYPE_F16: u8 = 1;
const DTYPE_Q8_0: u8 = 8;
const DTYPE_Q4_K: u8 = 12;
const DTYPE_Q5_K: u8 = 13;
const DTYPE_Q6_K: u8 = 14;

/// Tensor definition for building synthetic APR v2 binary data
struct TensorDef {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    data: Vec<u8>,
}

/// Build APR v2 binary data from metadata JSON and tensor definitions.
fn build_apr_v2(metadata_json: &str, tensors: &[TensorDef]) -> Vec<u8> {
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Build tensor index entries
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

/// Create F32 tensor data filled with a constant value
fn make_f32_data(num_elements: usize, value: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_elements * 4);
    for _ in 0..num_elements {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

/// Create F16 tensor data: each element is f16 encoding of `value`
fn make_f16_data(num_elements: usize, value: f32) -> Vec<u8> {
    let bits = half::f16::from_f32(value).to_bits();
    let mut data = Vec::with_capacity(num_elements * 2);
    for _ in 0..num_elements {
        data.extend_from_slice(&bits.to_le_bytes());
    }
    data
}

/// Create Q8_0 tensor data: 34 bytes per block (2 f16 scale + 32 i8 quants), 32 elements/block
fn make_q8_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 34);
    for _ in 0..num_blocks {
        // scale = f16(1.0)
        data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        // 32 quants, all 1
        data.extend_from_slice(&[1i8 as u8; 32]);
    }
    data
}

/// Create Q4_K tensor data: 144 bytes per block, 256 elements/block
fn make_q4k_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(256);
    // Each block: 144 bytes (12 scale bytes + 4 min bytes + 128 quant bytes)
    // Just fill with zeros — produces zeros when dequantized, but exercises the code path
    vec![0u8; num_blocks * 144]
}

/// Create Q6_K tensor data: 210 bytes per block, 256 elements/block
fn make_q6k_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(256);
    // Each block: 210 bytes
    vec![0u8; num_blocks * 210]
}

/// Standard metadata JSON for a tiny model
fn minimal_metadata(
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    intermediate: usize,
) -> String {
    format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {hidden},
        "num_hidden_layers": {layers},
        "num_attention_heads": {heads},
        "num_key_value_heads": {kv_heads},
        "vocab_size": {vocab},
        "intermediate_size": {intermediate},
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "context_length": 512
    }}"#
    )
}

/// Build a minimal tensor set using HF naming with a given dtype for the weight tensors.
/// Returns a TensorDef vec suitable for build_apr_v2.
///
/// embedding and lm_head use `embed_dtype` and `lm_head_dtype` respectively.
/// layer weight tensors use `weight_dtype`.
/// norm tensors always use F32.
fn make_hf_tensors(
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    embed_dtype: u8,
    weight_dtype: u8,
    lm_head_dtype: u8,
) -> Vec<TensorDef> {
    let head_dim = hidden / heads;
    let kv_dim = kv_heads * head_dim;
    let qkv_out = hidden + 2 * kv_dim;

    let make_data = |dtype: u8, num_elements: usize| -> Vec<u8> {
        match dtype {
            DTYPE_F16 => make_f16_data(num_elements, 0.01),
            DTYPE_Q8_0 => make_q8_0_data(num_elements),
            DTYPE_Q4_K | DTYPE_Q5_K => make_q4k_data(num_elements),
            DTYPE_Q6_K => make_q6k_data(num_elements),
            _ => make_f32_data(num_elements, 0.01), // F32 default
        }
    };

    vec![
        TensorDef {
            name: "model.embed_tokens.weight".into(),
            dtype: embed_dtype,
            dims: vec![vocab as u64, hidden as u64],
            data: make_data(embed_dtype, vocab * hidden),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, hidden as u64],
            data: make_data(weight_dtype, hidden * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![kv_dim as u64, hidden as u64],
            data: make_data(weight_dtype, kv_dim * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![kv_dim as u64, hidden as u64],
            data: make_data(weight_dtype, kv_dim * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, hidden as u64],
            data: make_data(weight_dtype, hidden * hidden),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_data(weight_dtype, intermediate * hidden),
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_data(weight_dtype, intermediate * hidden),
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_data(weight_dtype, hidden * intermediate),
        },
        TensorDef {
            name: "model.norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".into(),
            dtype: lm_head_dtype,
            dims: vec![vocab as u64, hidden as u64],
            data: make_data(lm_head_dtype, vocab * hidden),
        },
    ]
}

// ============================================================================
// F16 dtype branch coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_f16_embedding_and_lm_head() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F16,
        DTYPE_F32,
        DTYPE_F16,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "F16 embedding+lm_head: {}",
        result.unwrap_err()
    );

    let apr = result.unwrap();
    assert_eq!(apr.token_embedding.len(), vocab * hidden);
    assert_eq!(apr.lm_head_weight.len(), vocab * hidden);
    // F16(0.01) -> F32 should be approximately 0.01
    assert!((apr.token_embedding[0] - 0.01).abs() < 0.002);
}

// ============================================================================
// Q8_0 dtype branch coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_q8_0_weights() {
    let hidden = 32; // must be multiple of 32 for Q8_0 blocks
    let intermediate = 64;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q8_0,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q8_0 weights: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers.len(), 1);
    // Q8_0 dequantized weights should be 1.0 (scale=1.0, quant=1)
    assert!((apr.layers[0].qkv_weight[0] - 1.0).abs() < 0.01);
}

// ============================================================================
// Q4_K dtype branch coverage (flat path — dims divisible by 256)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_flat_weights() {
    let hidden = 256; // multiple of 256 for Q4_K flat path
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4_K flat: {}", result.unwrap_err());

    let apr = result.unwrap();
    // Q4K weights loaded — also check q4k_layers populated
    assert!(
        apr.q4k_layers.is_some(),
        "Q4K raw bytes should be extracted"
    );
    let q4k = apr.q4k_layers.as_ref().unwrap();
    assert_eq!(q4k.len(), 1);
    assert!(q4k[0].attn_q_weight.is_some());
    assert!(q4k[0].ffn_gate_weight.is_some());
}

// ============================================================================
// Q5_K dtype branch coverage (flat path — uses Q4_K dequant)
// ============================================================================

#[test]
fn test_from_apr_bytes_q5k_flat_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q5_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q5_K flat: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers.len(), 1);
    // Q5_K should also produce q4k raw bytes (dtype 13 accepted by get_q4k_raw_bytes)
    assert!(apr.q4k_layers.is_some());
}

// ============================================================================
// Q6_K dtype branch coverage (flat path)
// ============================================================================

#[test]
fn test_from_apr_bytes_q6k_flat_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    // Use Q6_K for down_proj specifically (tests q6k_ffn_down path)
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Override specific tensors to Q6_K
    for t in &mut tensors {
        if t.name.contains("down_proj") || t.name.contains("up_proj") || t.name.contains("v_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6_K flat: {}", result.unwrap_err());

    let apr = result.unwrap();
    // Q6K tensors should produce q4k_layers with q6k fields populated
    assert!(apr.q4k_layers.is_some());
    let q4k = apr.q4k_layers.as_ref().unwrap();
    assert!(q4k[0].ffn_down_weight_q6k.is_some());
    assert!(q4k[0].ffn_up_weight_q6k.is_some());
    assert!(q4k[0].attn_v_weight_q6k.is_some());
}

// ============================================================================
// Q4_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_perrow_path() {
    // Use hidden=128 (not multiple of 256) so 2D Q4_K tensors hit perrow path
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4_K perrow: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers.len(), 1);
    // 2D weights with dims[1]=128 hit perrow dequant path
    assert_eq!(
        apr.layers[0].qkv_weight.len(),
        hidden * (hidden + 2 * hidden)
    );
}

// ============================================================================
// Q6_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q6k_perrow_path() {
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Override down_proj to Q6_K with 2D dims where cols % 256 != 0
    for t in &mut tensors {
        if t.name.contains("down_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6_K perrow: {}", result.unwrap_err());
}

// ============================================================================
// Q5_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q5k_perrow_path() {
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q5_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q5_K perrow: {}", result.unwrap_err());
}

// ============================================================================
// Weight tying path (no lm_head.weight, uses embedding)
// ============================================================================

#[test]
fn test_from_apr_bytes_weight_tying_via_embed() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove lm_head.weight to trigger weight tying
    tensors.retain(|t| t.name != "lm_head.weight");
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Weight tying should succeed: {}",
        result.unwrap_err()
    );

    let apr = result.unwrap();
    // lm_head should be same as embedding
    assert_eq!(apr.lm_head_weight.len(), vocab * hidden);
}

// ============================================================================
// GGUF naming convention (blk.X, token_embd, output_norm, output.weight)
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_naming() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);

    let tensors = vec![
        TensorDef {
            name: "token_embd.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.1),
        },
        TensorDef {
            name: "blk.0.attn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.attn_q.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_k.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_v.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_output.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_gate.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.01),
        },
        TensorDef {
            name: "output_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "output.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
    ];
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GGUF naming: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers.len(), 1);
    assert!(apr.layers[0].ffn_norm_weight.is_some());
    assert!(apr.layers[0].ffn_gate_weight.is_some());
}

// ============================================================================
// Metadata alias coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_metadata_aliases() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    // Use alternative metadata field names to cover .or_else() branches
    let meta = r#"{
        "model_type": "qwen2",
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 4,
        "vocab_size": 16,
        "intermediate_dim": 32,
        "rms_norm_eps": 1e-6,
        "max_seq_len": 1024
    }"#;
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Metadata aliases: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.config.architecture, "qwen2");
    assert_eq!(apr.config.hidden_dim, hidden);
    assert_eq!(apr.config.intermediate_dim, intermediate);
    assert_eq!(apr.config.context_length, 1024);
}

#[test]
fn test_from_apr_bytes_architecture_auto_filtered() {
    let meta = r#"{
        "architecture": "Auto",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 16,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-6
    }"#;
    let tensors = make_hf_tensors(8, 32, 4, 4, 16, DTYPE_F32, DTYPE_F32, DTYPE_F32);
    let data = build_apr_v2(meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());
    // "Auto" should be filtered out and defaulted to "unknown"
    assert_eq!(result.unwrap().config.architecture, "unknown");
}

// ============================================================================
// Error paths
// ============================================================================

#[test]
fn test_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"NOPE");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Invalid APR magic"));
}

#[test]
fn test_from_apr_bytes_truncated_metadata() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    // Set metadata offset=64, size=9999 (beyond file)
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&9999u32.to_le_bytes());
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Metadata extends beyond file"));
}

#[test]
fn test_from_apr_bytes_no_embedding_tensor() {
    // Valid header + metadata but NO embedding tensor
    let meta = r#"{"hidden_size": 8, "num_hidden_layers": 0, "vocab_size": 4}"#;
    let tensors = vec![
        TensorDef {
            name: "model.norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![8],
            data: make_f32_data(8, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![4, 8],
            data: make_f32_data(32, 0.01),
        },
    ];
    let data = build_apr_v2(meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("No embedding tensor found"));
}

#[test]
fn test_from_apr_bytes_no_lm_head_no_embed_for_tying() {
    // Has token_embd.weight but no lm_head.weight — exercises weight tying via token_embd
    let meta = r#"{"hidden_size": 8, "num_hidden_layers": 0, "vocab_size": 4}"#;
    let tensors = vec![TensorDef {
        name: "token_embd.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![4, 8],
        data: make_f32_data(32, 0.1),
    }];
    let data = build_apr_v2(meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    // token_embd.weight should be found as embedding AND as tied lm_head
    assert!(result.is_ok(), "token_embd tying: {}", result.unwrap_err());
}

// ============================================================================
// Q4K lm_head paths (lm_head_weight_q4k / lm_head_weight_q6k)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256; // multiple of 256 for flat Q4K
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4K lm_head: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert!(apr.lm_head_weight_q4k.is_some());
}

#[test]
fn test_from_apr_bytes_q6k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Set lm_head to Q6_K
    for t in &mut tensors {
        if t.name == "lm_head.weight" {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6K lm_head: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert!(apr.lm_head_weight_q6k.is_some());
}

// ============================================================================
// Fused QKV bias path
// ============================================================================

#[test]
fn test_from_apr_bytes_fused_qkv_bias() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Add a fused QKV bias tensor
    let qkv_out = hidden + 2 * hidden; // heads=4, kv_heads=4 => kv_dim=hidden
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.0),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Fused QKV bias: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert!(apr.layers[0].qkv_bias.is_some());
    assert_eq!(apr.layers[0].qkv_bias.as_ref().unwrap().len(), qkv_out);
}

// ============================================================================
// Separate Q/K/V bias path
// ============================================================================

#[test]
fn test_from_apr_bytes_separate_qkv_biases() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let kv_dim = hidden; // kv_heads=4, head_dim=2
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Add separate Q/K/V biases
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.q_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![hidden as u64],
        data: make_f32_data(hidden, 0.1),
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.k_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![kv_dim as u64],
        data: make_f32_data(kv_dim, 0.2),
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.v_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![kv_dim as u64],
        data: make_f32_data(kv_dim, 0.3),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Separate QKV biases: {}",
        result.unwrap_err()
    );

    let apr = result.unwrap();
    let bias = apr.layers[0].qkv_bias.as_ref().unwrap();
    assert_eq!(bias.len(), hidden + kv_dim + kv_dim);
    // Q bias = 0.1
    assert!((bias[0] - 0.1).abs() < 1e-6);
    // K bias = 0.2
    assert!((bias[hidden] - 0.2).abs() < 1e-6);
    // V bias = 0.3
    assert!((bias[hidden + kv_dim] - 0.3).abs() < 1e-6);
}

// ============================================================================
// Fused QKV weight path (single combined tensor)
// ============================================================================

#[test]
fn test_from_apr_bytes_fused_qkv_weight() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let kv_dim = hidden; // kv_heads=4
    let qkv_out = hidden + 2 * kv_dim;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove separate Q/K/V and add fused QKV
    tensors.retain(|t| {
        !t.name.contains("q_proj.weight")
            && !t.name.contains("k_proj.weight")
            && !t.name.contains("v_proj.weight")
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64, hidden as u64],
        data: make_f32_data(qkv_out * hidden, 0.02),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Fused QKV: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers[0].qkv_weight.len(), qkv_out * hidden);
}

// ============================================================================
// GGUF naming with Q4K weights (tests get_q4k_raw_bytes with GGUF prefixes)
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_q4k_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);

    let tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.1),
        },
        TensorDef {
            name: "blk.0.attn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.attn_q.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_k.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_v.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_output.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_gate.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_q4k_data(intermediate * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_q4k_data(intermediate * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_q4k_data(hidden * intermediate),
        },
        TensorDef {
            name: "output_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "output.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![vocab as u64, hidden as u64],
            data: make_q4k_data(vocab * hidden),
        },
    ];
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GGUF Q4K: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert!(apr.q4k_layers.is_some());
    let q4k = apr.q4k_layers.as_ref().unwrap();
    assert!(q4k[0].attn_q_weight.is_some());
    assert!(q4k[0].attn_k_weight.is_some());
    assert!(q4k[0].attn_output_weight.is_some());
    assert!(q4k[0].ffn_gate_weight.is_some());
    assert!(q4k[0].ffn_up_weight.is_some());
    assert!(q4k[0].ffn_down_weight.is_some());
    assert!(apr.lm_head_weight_q4k.is_some());
}

// ============================================================================
// Multi-layer model
// ============================================================================

#[test]
fn test_from_apr_bytes_multi_layer() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let num_layers = 3;
    let meta = minimal_metadata(hidden, num_layers, 4, 4, vocab, intermediate);

    let mut tensors = vec![TensorDef {
        name: "model.embed_tokens.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![vocab as u64, hidden as u64],
        data: make_f32_data(vocab * hidden, 0.1),
    }];

    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");
        tensors.extend(vec![
            TensorDef {
                name: format!("{prefix}.input_layernorm.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64],
                data: make_f32_data(hidden, 1.0),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.q_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.k_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.v_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.o_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.post_attention_layernorm.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64],
                data: make_f32_data(hidden, 1.0),
            },
            TensorDef {
                name: format!("{prefix}.mlp.up_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![intermediate as u64, hidden as u64],
                data: make_f32_data(intermediate * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.mlp.down_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, intermediate as u64],
                data: make_f32_data(hidden * intermediate, 0.01),
            },
        ]);
    }

    tensors.push(TensorDef {
        name: "model.norm.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![hidden as u64],
        data: make_f32_data(hidden, 1.0),
    });
    tensors.push(TensorDef {
        name: "lm_head.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![vocab as u64, hidden as u64],
        data: make_f32_data(vocab * hidden, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Multi-layer: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.layers.len(), num_layers);
    assert_eq!(apr.config.num_layers, num_layers);
}

// ============================================================================
// GQA model (num_kv_heads < num_heads)
// ============================================================================

#[test]
fn test_from_apr_bytes_gqa_model() {
    let hidden = 16;
    let intermediate = 32;
    let vocab = 8;
    let heads = 4;
    let kv_heads = 2;
    let head_dim = hidden / heads; // 4
    let kv_dim = kv_heads * head_dim; // 8
    let meta = minimal_metadata(hidden, 1, heads, kv_heads, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        heads,
        kv_heads,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GQA: {}", result.unwrap_err());

    let apr = result.unwrap();
    assert_eq!(apr.config.num_kv_heads, kv_heads);
    // QKV weight should be hidden + 2*kv_dim = 16 + 2*8 = 32 rows * hidden=16 cols = 512
    assert_eq!(
        apr.layers[0].qkv_weight.len(),
        (hidden + 2 * kv_dim) * hidden
    );
}

// ============================================================================
// forward_with_cache coverage (AprTransformer::forward_with_cache)
// ============================================================================

/// Build a minimal executable F32 AprTransformer via from_apr_bytes
fn build_f32_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("F32 APR build failed")
}

/// Build a minimal executable Q4K AprTransformer (with q4k_layers)
fn build_q4k_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("Q4K APR build failed")
}

use crate::apr_transformer::AprKVCache;

#[test]
fn test_forward_with_cache_f32_first_token() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token — exercises cache_len == 0 path (V used directly)
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "F32 cache first token: {}",
        result.unwrap_err()
    );
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16); // vocab_size
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_with_cache_f32_multi_token() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token (cache_len == 0)
    let _ = apr.forward_with_cache(1, &mut cache, 0).unwrap();

    // Second token — exercises cache_len > 0 path (full attention with cache)
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "F32 cache second token: {}",
        result.unwrap_err()
    );
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_with_cache_f32_three_tokens() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // Multiple tokens to exercise the full attention score computation
    for pos in 0..3 {
        let result = apr.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(
            result.is_ok(),
            "F32 cache token {pos}: {}",
            result.unwrap_err()
        );
    }
}

#[test]
fn test_forward_with_cache_q4k_first_token() {
    // Q4K weights exercise the fused kernel paths in forward_with_cache
    let apr = build_q4k_apr(256, 256, 4);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K cache first token: {}",
        result.unwrap_err()
    );
    let logits = result.unwrap();
    assert_eq!(logits.len(), 4);
}

#[test]
fn test_forward_with_cache_q4k_multi_token() {
    let apr = build_q4k_apr(256, 256, 4);
    let mut cache = AprKVCache::new(&apr.config);

    // Exercise both cache_len == 0 and cache_len > 0 paths with Q4K
    let _ = apr.forward_with_cache(1, &mut cache, 0).unwrap();
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q4K cache second token: {}",
        result.unwrap_err()
    );
}

#[test]
fn test_forward_with_cache_q6k_weights() {
    // Q6K weights for v_proj, ffn_down, ffn_up exercise the Q6K matmul paths
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    // Override specific tensors to Q6_K
    for t in &mut tensors {
        if t.name.contains("v_proj") || t.name.contains("down_proj") || t.name.contains("up_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q6K APR build failed");
    let mut cache = AprKVCache::new(&apr.config);

    // First token exercises Q6K V projection and Q6K FFN paths
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q6K cache first token: {}",
        result.unwrap_err()
    );
}

#[test]
fn test_forward_with_cache_with_qkv_bias() {
    // Model with QKV biases exercises the bias-add paths in forward_with_cache
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let kv_dim = hidden; // kv_heads=4, head_dim=2
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Add fused QKV bias
    let qkv_out = hidden + 2 * kv_dim;
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("biased APR build failed");
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Biased cache: {}", result.unwrap_err());
}

// ============================================================================
// AprTransformer::forward coverage (non-cached full sequence)
// ============================================================================

#[test]
fn test_forward_f32_single_token() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Forward single: {}", result.unwrap_err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_f32_multi_token() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Forward multi: {}", result.unwrap_err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_q4k_single_token() {
    let apr = build_q4k_apr(256, 256, 4);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K forward: {}", result.unwrap_err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 4);
}

// ============================================================================
// AprTransformer::generate coverage
// ============================================================================

#[test]
fn test_generate_f32_greedy() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.generate(&[1], 3);
    assert!(result.is_ok(), "Generate: {}", result.unwrap_err());
    let tokens = result.unwrap();
    assert!(tokens.len() >= 2); // at least prompt + 1 generated
    assert!(tokens.len() <= 4); // prompt + max 3 tokens
    assert_eq!(tokens[0], 1); // prompt preserved
}

// ============================================================================
// AprTransformer helper functions coverage
// ============================================================================

#[test]
fn test_apr_transformer_config_accessor() {
    let apr = build_f32_apr(8, 32, 16);
    let config = apr.config();
    assert_eq!(config.hidden_dim, 8);
    assert_eq!(config.intermediate_dim, 32);
    assert_eq!(config.vocab_size, 16);
}

#[test]
fn test_apr_transformer_num_parameters() {
    let apr = build_f32_apr(8, 32, 16);
    let params = apr.num_parameters();
    // Should be > 0
    assert!(params > 0);
    // Rough check: at minimum vocab*hidden*2 (embed + lm_head)
    assert!(params >= 16 * 8 * 2);
}

// ============================================================================
// Standard MLP (GELU, no gate) path coverage
// ============================================================================

/// Build a model WITHOUT ffn_gate_weight (GELU/standard MLP path, like phi-2)
fn build_gelu_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove gate_proj to make the model use standard MLP (GELU) instead of SwiGLU
    tensors.retain(|t| !t.name.contains("gate_proj"));
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("GELU APR build failed")
}

#[test]
fn test_forward_gelu_model_single_token() {
    let apr = build_gelu_apr(8, 32, 16);
    // Verify no gate weight
    assert!(apr.layers[0].ffn_gate_weight.is_none());
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "GELU forward: {}", result.unwrap_err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_with_cache_gelu_model() {
    let apr = build_gelu_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token (GELU + cache_len == 0)
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "GELU cache first: {}", result.unwrap_err());

    // Second token (GELU + cache_len > 0)
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(result.is_ok(), "GELU cache second: {}", result.unwrap_err());
}

// ============================================================================
// No ffn_norm path (hidden passed directly to FFN)
// ============================================================================

#[test]
fn test_forward_with_cache_no_ffn_norm() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove post_attention_layernorm to exercise no-ffn-norm path
    tensors.retain(|t| !t.name.contains("post_attention_layernorm"));
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("no-ffn-norm APR build failed");
    assert!(apr.layers[0].ffn_norm_weight.is_none());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "No FFN norm: {}", result.unwrap_err());
}

// ============================================================================
// Q4K fused kernel with QKV bias in forward_with_cache
// ============================================================================

#[test]
fn test_forward_with_cache_q4k_with_bias() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let kv_dim = hidden; // kv_heads=4
    let qkv_out = hidden + 2 * kv_dim;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    // Add QKV bias (exercises the bias-split path when q4k_layer.is_some())
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q4K+bias APR build failed");
    assert!(apr.q4k_layers.is_some());
    assert!(apr.layers[0].qkv_bias.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K+bias cache: {}", result.unwrap_err());
}

// ============================================================================
// Q4K lm_head in forward_with_cache
// ============================================================================

#[test]
fn test_forward_with_cache_q4k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q4K lm_head APR build failed");
    assert!(apr.lm_head_weight_q4k.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K lm_head cache: {}", result.unwrap_err());
}

#[test]
fn test_forward_with_cache_q6k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Set lm_head to Q6_K
    for t in &mut tensors {
        if t.name == "lm_head.weight" {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q6K lm_head APR build failed");
    assert!(apr.lm_head_weight_q6k.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K lm_head cache: {}", result.unwrap_err());
}
