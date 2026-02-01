//! APR Test Factory - Active Pygmy Builder
//!
//! Provides `build_executable_pygmy_apr()` for testing APR inference paths.

use super::{HEADER_SIZE, MAGIC};

/// Build an **executable** Pygmy APR model (F32 tensors, minimal dimensions)
///
/// Creates a minimal APR v2 model with:
/// - hidden_size: 8
/// - num_layers: 1
/// - num_heads: 2
/// - num_kv_heads: 2
/// - vocab_size: 10
/// - intermediate_size: 16
///
/// All tensors use F32 format for simplicity.
pub fn build_executable_pygmy_apr() -> Vec<u8> {
    let hidden_size = 8;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 10;
    let intermediate_size = 16;

    let metadata = format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {hidden_size},
        "num_layers": {num_layers},
        "num_heads": {num_heads},
        "num_kv_heads": {num_kv_heads},
        "vocab_size": {vocab_size},
        "intermediate_size": {intermediate_size},
        "rms_norm_eps": 1e-6
    }}"#
    );
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Tensors needed for forward pass (all F32):
    let tensor_defs: Vec<(&str, Vec<u64>, usize)> = vec![
        (
            "model.embed_tokens.weight",
            vec![vocab_size as u64, hidden_size as u64],
            vocab_size * hidden_size * 4,
        ),
        (
            "layers.0.input_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "layers.0.self_attn.q_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.post_attention_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "layers.0.mlp.down_proj.weight",
            vec![hidden_size as u64, intermediate_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        ("norm.weight", vec![hidden_size as u64], hidden_size * 4),
        (
            "lm_head.weight",
            vec![vocab_size as u64, hidden_size as u64],
            vocab_size * hidden_size * 4,
        ),
    ];

    // Build tensor index entries
    let mut tensor_entries = Vec::new();
    let mut current_offset = 0u64;

    for (name, shape, byte_size) in &tensor_defs {
        let entry = create_tensor_entry(name, 0, shape, current_offset, *byte_size as u64);
        tensor_entries.push(entry);
        current_offset += *byte_size as u64;
    }

    let tensor_index: Vec<u8> = tensor_entries
        .iter()
        .flat_map(|e| e.iter().copied())
        .collect();
    let tensor_count = tensor_defs.len() as u32;
    let total_data_size = current_offset as usize;

    // Calculate offsets
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let total_size = data_offset as usize + total_data_size;

    let mut data = vec![0u8; total_size];

    // Write header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Write tensor data with small values
    let data_start = data_offset as usize;
    let num_floats = total_data_size / 4;
    for i in 0..num_floats {
        let val = ((i % 10) as f32 - 5.0) * 0.1;
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    // Set layernorm weights to 1.0
    let norm_offsets = vec![
        vocab_size * hidden_size * 4,
        vocab_size * hidden_size * 4 + hidden_size * 4 + 4 * hidden_size * hidden_size * 4,
        vocab_size * hidden_size * 4
            + hidden_size * 4
            + 4 * hidden_size * hidden_size * 4
            + hidden_size * 4
            + 3 * hidden_size * intermediate_size * 4,
    ];

    for offset in norm_offsets {
        for i in 0..hidden_size {
            let val = 1.0f32;
            let pos = data_start + offset + i * 4;
            if pos + 4 <= data.len() {
                data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }

    data
}

/// Create a binary tensor entry for the APR v2 tensor index (variable-size format)
fn create_tensor_entry(name: &str, dtype: u8, shape: &[u64], offset: u64, size: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(name.len() as u16).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.push(dtype);
    data.push(shape.len() as u8);
    for &dim in shape {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    data
}

/// Build a Pygmy APR model using GGUF naming conventions
///
/// GGUF uses different tensor names than HuggingFace:
/// - `token_embd.weight` instead of `model.embed_tokens.weight`
/// - `blk.N.attn_q.weight` instead of `layers.N.self_attn.q_proj.weight`
/// - `output_norm.weight` instead of `norm.weight`
/// - Weight tying: no separate `lm_head.weight`, use `token_embd.weight` transposed
///
/// GH-194: This tests the weight tying path that was missing `token_embd.weight` lookup.
pub fn build_executable_pygmy_apr_gguf_names() -> Vec<u8> {
    let hidden_size = 8;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 10;
    let intermediate_size = 16;

    let metadata = format!(
        r#"{{
        "architecture": "qwen2",
        "hidden_size": {hidden_size},
        "num_layers": {num_layers},
        "num_heads": {num_heads},
        "num_kv_heads": {num_kv_heads},
        "vocab_size": {vocab_size},
        "intermediate_size": {intermediate_size},
        "rms_norm_eps": 1e-6
    }}"#
    );
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // GGUF tensor naming convention - NO separate lm_head (weight tying)
    // GH-194: token_embd.weight is used for both embedding and lm_head
    let tensor_defs: Vec<(&str, Vec<u64>, usize)> = vec![
        // Embedding table - will be used for both embed and lm_head (weight tying)
        (
            "token_embd.weight",
            vec![hidden_size as u64, vocab_size as u64], // [hidden, vocab] for tied weights
            vocab_size * hidden_size * 4,
        ),
        // Layer 0 tensors with GGUF naming
        (
            "blk.0.attn_norm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "blk.0.attn_q.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "blk.0.attn_k.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "blk.0.attn_v.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "blk.0.attn_output.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![hidden_size as u64, intermediate_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "blk.0.ffn_up.weight",
            vec![hidden_size as u64, intermediate_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "blk.0.ffn_down.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        // Final norm - GGUF naming
        (
            "output_norm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        // NO lm_head.weight - weight tying uses token_embd.weight
    ];

    build_apr_from_tensor_defs(metadata_bytes, metadata_padded_size, &tensor_defs)
}

/// Build a Pygmy APR with explicit weight tying (embed_tokens.weight used as lm_head)
///
/// This simulates a model where the embedding table is explicitly tied to the
/// output projection. The lm_head lookup should find `embed_tokens.weight`.
pub fn build_executable_pygmy_apr_embed_tied() -> Vec<u8> {
    let hidden_size = 8;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 10;
    let intermediate_size = 16;

    let metadata = format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {hidden_size},
        "num_layers": {num_layers},
        "num_heads": {num_heads},
        "num_kv_heads": {num_kv_heads},
        "vocab_size": {vocab_size},
        "intermediate_size": {intermediate_size},
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true
    }}"#
    );
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // HuggingFace naming but with weight tying (no lm_head)
    let tensor_defs: Vec<(&str, Vec<u64>, usize)> = vec![
        (
            "embed_tokens.weight",
            vec![vocab_size as u64, hidden_size as u64],
            vocab_size * hidden_size * 4,
        ),
        (
            "layers.0.input_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "layers.0.self_attn.q_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            vec![hidden_size as u64, hidden_size as u64],
            hidden_size * hidden_size * 4,
        ),
        (
            "layers.0.post_attention_layernorm.weight",
            vec![hidden_size as u64],
            hidden_size * 4,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            vec![intermediate_size as u64, hidden_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        (
            "layers.0.mlp.down_proj.weight",
            vec![hidden_size as u64, intermediate_size as u64],
            hidden_size * intermediate_size * 4,
        ),
        ("norm.weight", vec![hidden_size as u64], hidden_size * 4),
        // NO lm_head.weight - uses embed_tokens.weight (tied)
    ];

    build_apr_from_tensor_defs(metadata_bytes, metadata_padded_size, &tensor_defs)
}

/// Common APR builder from tensor definitions
fn build_apr_from_tensor_defs(
    metadata_bytes: &[u8],
    metadata_padded_size: usize,
    tensor_defs: &[(&str, Vec<u64>, usize)],
) -> Vec<u8> {
    // Build tensor index entries
    let mut tensor_entries = Vec::new();
    let mut current_offset = 0u64;

    for (name, shape, byte_size) in tensor_defs {
        let entry = create_tensor_entry(name, 0, shape, current_offset, *byte_size as u64);
        tensor_entries.push(entry);
        current_offset += *byte_size as u64;
    }

    let tensor_index: Vec<u8> = tensor_entries
        .iter()
        .flat_map(|e| e.iter().copied())
        .collect();
    let tensor_count = tensor_defs.len() as u32;
    let total_data_size = current_offset as usize;

    // Calculate offsets
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let total_size = data_offset as usize + total_data_size;

    let mut data = vec![0u8; total_size];

    // Write header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Write tensor data with small non-zero values
    let data_start = data_offset as usize;
    let num_floats = total_data_size / 4;
    for i in 0..num_floats {
        let val = ((i % 10) as f32 - 5.0) * 0.01; // Small values for stability
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    // Set norm weights to 1.0 (required for RMS norm to work)
    // Find and set all norm tensor values to 1.0
    let mut offset = 0usize;
    for (name, _shape, byte_size) in tensor_defs {
        if name.contains("norm") {
            let norm_start = data_start + offset;
            let norm_count = byte_size / 4;
            for i in 0..norm_count {
                let pos = norm_start + i * 4;
                if pos + 4 <= data.len() {
                    data[pos..pos + 4].copy_from_slice(&1.0f32.to_le_bytes());
                }
            }
        }
        offset += byte_size;
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr::AprV2Model;

    #[test]
    fn test_factory_builds_valid_apr() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data);
        assert!(model.is_ok());
    }

    #[test]
    fn test_factory_apr_forward_works() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).unwrap();
        let logits = model.forward(&[1]);
        assert!(logits.is_ok());
    }

    // =========================================================================
    // GH-194: Weight Tying Tests (token_embd.weight as lm_head)
    // =========================================================================

    /// GH-194: GGUF naming convention model must load successfully
    #[test]
    fn test_gh194_gguf_names_model_loads() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data);
        assert!(model.is_ok(), "GH-194: GGUF-named model must load");
    }

    /// GH-194: GGUF model with token_embd.weight (no lm_head) must find lm_head tensor
    #[test]
    fn test_gh194_gguf_names_finds_lm_head_via_token_embd() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data).unwrap();

        // The model should find token_embd.weight when looking for lm_head
        let lm_head_candidates = [
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
        ];

        let found = lm_head_candidates
            .iter()
            .any(|&name| model.get_tensor(name).is_some());

        assert!(
            found,
            "GH-194: Model must find at least one lm_head candidate. \
             token_embd.weight should be found for weight tying."
        );
    }

    /// GH-194: GGUF model forward pass must work with weight tying
    #[test]
    fn test_gh194_gguf_names_forward_works() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data).unwrap();

        // Forward should work even without explicit lm_head.weight
        let result = model.forward(&[1]);
        assert!(
            result.is_ok(),
            "GH-194: Forward must work with token_embd.weight as lm_head. Error: {:?}",
            result.err()
        );

        // Logits should have vocab_size elements
        let logits = result.unwrap();
        assert_eq!(
            logits.len(),
            10,
            "GH-194: Logits must have vocab_size=10 elements"
        );
    }

    /// GH-194: embed_tokens.weight tied model must work
    #[test]
    fn test_gh194_embed_tied_forward_works() {
        let data = build_executable_pygmy_apr_embed_tied();
        let model = AprV2Model::from_bytes(data).unwrap();

        let result = model.forward(&[1]);
        assert!(
            result.is_ok(),
            "GH-194: Forward must work with embed_tokens.weight as lm_head. Error: {:?}",
            result.err()
        );
    }

    /// GH-194: Tensor count must match expected count (no silent drops)
    #[test]
    fn test_gh194_tensor_count_preserved() {
        // HuggingFace naming: 12 tensors
        let hf_data = build_executable_pygmy_apr();
        let hf_model = AprV2Model::from_bytes(hf_data).unwrap();
        assert_eq!(
            hf_model.tensor_count(),
            12,
            "HuggingFace model must have 12 tensors"
        );

        // GGUF naming: 11 tensors (no lm_head due to weight tying)
        let gguf_data = build_executable_pygmy_apr_gguf_names();
        let gguf_model = AprV2Model::from_bytes(gguf_data).unwrap();
        assert_eq!(
            gguf_model.tensor_count(),
            11,
            "GGUF model must have 11 tensors (weight tying)"
        );

        // Embed-tied naming: 11 tensors (no lm_head)
        let tied_data = build_executable_pygmy_apr_embed_tied();
        let tied_model = AprV2Model::from_bytes(tied_data).unwrap();
        assert_eq!(
            tied_model.tensor_count(),
            11,
            "Embed-tied model must have 11 tensors"
        );
    }

    /// GH-194: All tensor naming conventions must produce valid logits
    #[test]
    fn test_gh194_all_naming_conventions_produce_valid_logits() {
        let models = vec![
            ("HuggingFace", build_executable_pygmy_apr()),
            ("GGUF", build_executable_pygmy_apr_gguf_names()),
            ("EmbedTied", build_executable_pygmy_apr_embed_tied()),
        ];

        for (name, data) in models {
            let model =
                AprV2Model::from_bytes(data).unwrap_or_else(|_| panic!("{name} model must load"));
            let logits = model
                .forward(&[1])
                .unwrap_or_else(|_| panic!("{name} forward must work"));

            // Logits must be valid floats (no NaN/Inf)
            assert!(
                logits.iter().all(|&x| x.is_finite()),
                "{name}: Logits must be finite (no NaN/Inf)"
            );

            // Logits must have correct size
            assert_eq!(logits.len(), 10, "{name}: Logits must have vocab_size=10");
        }
    }
}
