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
}
