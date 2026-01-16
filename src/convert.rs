//! GGUF to APR Transformer Converter
//!
//! Converts GGUF models to APR Transformer format for fair comparison.
//! All weights are dequantized to F32 for WASM compatibility.
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::convert::GgufToAprConverter;
//!
//! let gguf_data = std::fs::read("model.gguf")?;
//! let apr_transformer = GgufToAprConverter::convert(&gguf_data)?;
//!
//! // Save to APR format
//! let apr_bytes = apr_transformer.to_apr_bytes()?;
//! std::fs::write("model.apr_transformer", apr_bytes)?;
//! ```

use crate::apr::{AprHeader, TensorEntry, ALIGNMENT, HEADER_SIZE, MAGIC};
use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFModel, GGUFTransformer};

/// GGUF to APR Transformer converter
///
/// Converts GGUF models with quantized weights to APR format with F32 weights.
/// This enables fair comparison between GGUF and APR serving performance.
pub struct GgufToAprConverter;

impl GgufToAprConverter {
    /// Convert GGUF file bytes to APR Transformer
    ///
    /// # Arguments
    ///
    /// * `gguf_data` - Raw GGUF file bytes
    ///
    /// # Returns
    ///
    /// `AprTransformer` with dequantized F32 weights
    ///
    /// # Errors
    ///
    /// Returns error if GGUF parsing or conversion fails
    pub fn convert(gguf_data: &[u8]) -> Result<AprTransformer> {
        // Parse GGUF model
        let gguf_model = GGUFModel::from_bytes(gguf_data)?;

        // Load transformer weights (dequantizes to F32)
        let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, gguf_data)?;

        // Convert to APR format
        Ok(Self::from_gguf_transformer(&gguf_transformer))
    }

    /// Convert from existing `GGUFTransformer` to `AprTransformer`
    ///
    /// # Arguments
    ///
    /// * `gguf` - Loaded GGUF transformer with dequantized weights
    ///
    /// # Returns
    ///
    /// `AprTransformer` with the same weights
    pub fn from_gguf_transformer(gguf: &GGUFTransformer) -> AprTransformer {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
        }
    }

    /// Convert APR Transformer to serialized APR v2 bytes
    ///
    /// Creates a valid .apr v2 file with:
    /// - APR v2 header (64 bytes)
    /// - JSON metadata (padded to 64-byte boundary)
    /// - Tensor index (JSON array)
    /// - Tensor data (each 64-byte aligned)
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer to serialize
    ///
    /// # Returns
    ///
    /// Raw bytes in APR v2 format
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_apr_bytes(transformer: &AprTransformer) -> Result<Vec<u8>> {
        // Serialize metadata
        let metadata = serde_json::json!({
            "model_type": "transformer_lm",
            "architecture": transformer.config.architecture,
            "hidden_size": transformer.config.hidden_dim,
            "num_layers": transformer.config.num_layers,
            "num_heads": transformer.config.num_heads,
            "num_kv_heads": transformer.config.num_kv_heads,
            "vocab_size": transformer.config.vocab_size,
            "intermediate_dim": transformer.config.intermediate_dim,
            "context_length": transformer.config.context_length,
            "rope_theta": transformer.config.rope_theta,
            "eps": transformer.config.eps,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize metadata: {e}"),
            })?;

        // Pad metadata to 64-byte boundary
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Serialize weights as single tensor (JSON payload for now)
        let payload_bytes =
            serde_json::to_vec(transformer).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize weights: {e}"),
            })?;

        // Create tensor index with single entry for the full payload
        let tensor_entries = vec![TensorEntry {
            name: "weights".to_string(),
            dtype: "json".to_string(),
            shape: vec![payload_bytes.len()],
            offset: 0,
            size: payload_bytes.len() as u64,
        }];
        let tensor_index_bytes =
            serde_json::to_vec(&tensor_entries).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize tensor index: {e}"),
            })?;

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

        // Build APR v2 header (64 bytes)
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
        header[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset.to_le_bytes());
        header[40..44].copy_from_slice(&0u32.to_le_bytes()); // checksum (TODO)
                                                             // bytes 44-63 reserved

        // Combine all parts
        let total_size =
            HEADER_SIZE + metadata_padded_len + tensor_index_bytes.len() + payload_bytes.len();
        let mut result = Vec::with_capacity(total_size);
        result.extend_from_slice(&header);
        result.extend_from_slice(&metadata_bytes);
        result.resize(HEADER_SIZE + metadata_padded_len, 0); // pad metadata
        result.extend_from_slice(&tensor_index_bytes);
        result.extend_from_slice(&payload_bytes);

        Ok(result)
    }

    /// Load APR Transformer from APR v2 bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw APR v2 file bytes
    ///
    /// # Returns
    ///
    /// Loaded `AprTransformer`
    ///
    /// # Errors
    ///
    /// Returns error if parsing fails
    pub fn from_apr_bytes(data: &[u8]) -> Result<AprTransformer> {
        // Parse header
        let header = AprHeader::from_bytes(data)?;

        // Get tensor index to find the weights tensor
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        if data.len() < index_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file truncated: expected {} bytes for tensor index, got {}",
                    index_end,
                    data.len()
                ),
            });
        }

        let tensor_entries: Vec<TensorEntry> =
            serde_json::from_slice(&data[index_start..index_end]).map_err(|e| {
                RealizarError::FormatError {
                    reason: format!("Failed to parse tensor index: {e}"),
                }
            })?;

        // Find the weights tensor
        let weights_entry = tensor_entries
            .iter()
            .find(|e| e.name == "weights")
            .ok_or_else(|| RealizarError::FormatError {
                reason: "No 'weights' tensor found in APR file".to_string(),
            })?;

        // Extract weights data
        let data_start = header.data_offset as usize + weights_entry.offset as usize;
        let data_end = data_start + weights_entry.size as usize;

        if data.len() < data_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file truncated: expected {} bytes for tensor data, got {}",
                    data_end,
                    data.len()
                ),
            });
        }

        let payload_bytes = &data[data_start..data_end];

        // Deserialize transformer
        let transformer: AprTransformer =
            serde_json::from_slice(payload_bytes).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to deserialize transformer: {e}"),
            })?;

        Ok(transformer)
    }

    /// Get conversion statistics
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer to analyze
    ///
    /// # Returns
    ///
    /// Statistics about the conversion
    pub fn stats(transformer: &AprTransformer) -> ConversionStats {
        let params = transformer.num_parameters();
        let memory_bytes = transformer.memory_size();

        ConversionStats {
            total_parameters: params,
            memory_bytes_f32: memory_bytes,
            num_layers: transformer.config.num_layers,
            hidden_dim: transformer.config.hidden_dim,
            vocab_size: transformer.config.vocab_size,
            architecture: transformer.config.architecture.clone(),
        }
    }
}

/// Statistics about a converted model
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Total number of parameters
    pub total_parameters: usize,
    /// Memory size in bytes (F32)
    pub memory_bytes_f32: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model architecture name
    pub architecture: String,
}

impl ConversionStats {
    /// Memory size in MB
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes_f32 as f64 / (1024.0 * 1024.0)
    }

    /// Memory size in GB
    #[must_use]
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes_f32 as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Parameters in millions
    #[must_use]
    pub fn parameters_m(&self) -> f64 {
        self.total_parameters as f64 / 1_000_000.0
    }

    /// Parameters in billions
    #[must_use]
    pub fn parameters_b(&self) -> f64 {
        self.total_parameters as f64 / 1_000_000_000.0
    }
}

// =============================================================================
// Q4K APR Converter (preserves GGUF quantization for GPU inference)
// =============================================================================

/// Raw tensor with preserved quantization
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Tensor name
    pub name: String,
    /// Raw bytes (Q4K super-blocks or F32/F16 data)
    pub data: Vec<u8>,
    /// Tensor shape (logical elements, not bytes)
    pub shape: Vec<usize>,
    /// GGML dtype: 0=F32, 1=F16, 6=Q4_K, 7=Q5_K, 8=Q8_0, 12=Q6_K
    pub dtype: u32,
}

/// GGUF to APR Q4K converter (preserves quantization)
///
/// Unlike `GgufToAprConverter` which dequantizes to F32, this converter
/// preserves Q4K/Q6K quantization for GPU inference with batched GEMV.
///
/// This is essential for achieving 2X Ollama performance.
pub struct GgufToAprQ4KConverter;

impl GgufToAprQ4KConverter {
    /// Helper to extract string from GGUF metadata
    fn get_string(metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Helper to extract u32 from GGUF metadata
    fn get_u32(metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>, key: &str) -> Option<u32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::UInt32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Int32(v)) => Some(*v as u32),
            Some(crate::gguf::GGUFValue::UInt64(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// Helper to extract f32 from GGUF metadata
    fn get_f32(metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>, key: &str) -> Option<f32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::Float32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// Convert GGUF file to APR v2 with preserved Q4K quantization
    ///
    /// # Arguments
    ///
    /// * `gguf_path` - Path to GGUF file
    /// * `output_path` - Path to write APR v2 file
    ///
    /// # Returns
    ///
    /// Statistics about the conversion
    #[allow(clippy::cast_possible_truncation)]
    pub fn convert(gguf_path: &std::path::Path, output_path: &std::path::Path) -> Result<Q4KConversionStats> {
        use std::io::Write;

        // Load GGUF with raw quantized tensors
        let gguf_data = std::fs::read(gguf_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read GGUF: {e}"),
        })?;

        let gguf_model = crate::gguf::GGUFModel::from_bytes(&gguf_data)?;

        // Extract model config from metadata
        let architecture = Self::get_string(&gguf_model.metadata, "general.architecture")
            .unwrap_or_else(|| "unknown".to_string());
        let hidden_size = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.embedding_length"))
            .unwrap_or(0);
        let num_layers = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.block_count"))
            .unwrap_or(0);
        let num_heads = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.attention.head_count"))
            .unwrap_or(0);
        let num_kv_heads = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.attention.head_count_kv"))
            .unwrap_or(num_heads);
        let vocab_size = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.vocab_size"))
            .or_else(|| Self::get_u32(&gguf_model.metadata, "tokenizer.ggml.vocab_size"))
            .unwrap_or_else(|| {
                // Infer from embedding tensor shape if metadata not available
                gguf_model.tensors.iter()
                    .find(|t| t.name.contains("token_embd") || t.name.contains("embed_tokens") || t.name.contains("tok_embeddings"))
                    .and_then(|t| t.dims.first().copied().map(|d| d as u32))
                    .unwrap_or(0)
            }) as usize;
        let intermediate_size = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.feed_forward_length"))
            .unwrap_or(0);
        let context_length = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.context_length"))
            .unwrap_or(2048);
        let rope_theta = Self::get_f32(&gguf_model.metadata, &format!("{architecture}.rope.freq_base"))
            .unwrap_or(10000.0);
        let eps = Self::get_f32(&gguf_model.metadata, &format!("{architecture}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        // Build metadata JSON
        let metadata = serde_json::json!({
            "model_type": "transformer_lm_q4k",
            "architecture": architecture,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "vocab_size": vocab_size,
            "intermediate_dim": intermediate_size,
            "context_length": context_length,
            "rope_theta": rope_theta,
            "eps": eps,
            "quantization": "Q4_K_M",
        });
        let metadata_bytes = serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
            reason: format!("Failed to serialize metadata: {e}"),
        })?;
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Extract raw tensors from GGUF
        let mut raw_tensors: Vec<RawTensor> = Vec::new();
        let mut q4k_count = 0usize;
        let mut total_bytes = 0usize;

        for tensor_meta in &gguf_model.tensors {
            let name = tensor_meta.name.clone();
            let shape: Vec<usize> = tensor_meta.dims.iter().map(|&d| d as usize).collect();
            let num_elements: usize = shape.iter().product();
            let qtype = tensor_meta.qtype;

            // Calculate byte size based on qtype (GGML dtype)
            // GGML types: 0=F32, 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
            let byte_size = match qtype {
                0 => num_elements * 4,      // F32
                1 => num_elements * 2,      // F16
                8 => (num_elements / 32) * 34, // Q8_0: 32 elements = 2 (scale) + 32 (quants)
                12 => (num_elements / 256) * 144, // Q4_K: 256 elements = 144 bytes
                13 => (num_elements / 256) * 176, // Q5_K: 256 elements = 176 bytes
                14 => (num_elements / 256) * 210, // Q6_K: 256 elements = 210 bytes
                _ => num_elements * 4,      // Default to F32
            };

            // Extract raw bytes
            let tensor_start = gguf_model.tensor_data_start + tensor_meta.offset as usize;
            if tensor_start + byte_size > gguf_data.len() {
                return Err(RealizarError::FormatError {
                    reason: format!("Tensor '{}' exceeds file bounds (start={}, size={}, file_len={})",
                        name, tensor_start, byte_size, gguf_data.len()),
                });
            }

            let data = gguf_data[tensor_start..tensor_start + byte_size].to_vec();

            // Q4_K is GGML type 12
            if qtype == 12 {
                q4k_count += 1;
            }
            total_bytes += byte_size;

            raw_tensors.push(RawTensor { name, data, shape, dtype: qtype });
        }

        // Build binary tensor index
        let mut tensor_index_bytes: Vec<u8> = Vec::new();
        let mut current_offset = 0u64;

        for tensor in &raw_tensors {
            // name_len (2 bytes) + name
            let name_bytes = tensor.name.as_bytes();
            tensor_index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            tensor_index_bytes.extend_from_slice(name_bytes);

            // dtype (1 byte) - map GGML dtype to APR dtype
            // GGML: 0=F32, 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
            // APR:  0=F32, 1=F16, 8=Q4_K, 9=Q6_K, 10=Q8_0
            let apr_dtype = match tensor.dtype {
                0 => 0u8,   // F32 -> F32
                1 => 1u8,   // F16 -> F16
                8 => 10u8,  // Q8_0 -> APR dtype 10
                12 => 8u8,  // Q4_K -> APR dtype 8
                13 => 8u8,  // Q5_K -> treat as Q4_K for now
                14 => 9u8,  // Q6_K -> APR dtype 9
                _ => 0u8,
            };
            tensor_index_bytes.push(apr_dtype);

            // ndim (1 byte) + dims (8 bytes each)
            tensor_index_bytes.push(tensor.shape.len() as u8);
            for &dim in &tensor.shape {
                tensor_index_bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // offset (8 bytes)
            tensor_index_bytes.extend_from_slice(&current_offset.to_le_bytes());

            // size (8 bytes)
            let size = tensor.data.len() as u64;
            tensor_index_bytes.extend_from_slice(&size.to_le_bytes());

            // Align next tensor to 64 bytes
            current_offset += size;
            let aligned = current_offset.div_ceil(ALIGNMENT as u64) * ALIGNMENT as u64;
            current_offset = aligned;
        }

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;
        // Align data offset
        let data_offset_aligned = data_offset.div_ceil(ALIGNMENT as u64) * ALIGNMENT as u64;

        // Build header (64 bytes)
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0x0020u16.to_le_bytes()); // flags: QUANTIZED=0x0020
        header[8..12].copy_from_slice(&(raw_tensors.len() as u32).to_le_bytes());
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset_aligned.to_le_bytes());

        // Write file
        let mut file = std::fs::File::create(output_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to create output file: {e}"),
        })?;

        // Header
        file.write_all(&header).map_err(|e| RealizarError::IoError {
            message: format!("Failed to write header: {e}"),
        })?;

        // Metadata (padded)
        file.write_all(&metadata_bytes).map_err(|e| RealizarError::IoError {
            message: format!("Failed to write metadata: {e}"),
        })?;
        let padding = metadata_padded_len - metadata_bytes.len();
        if padding > 0 {
            file.write_all(&vec![0u8; padding]).map_err(|e| RealizarError::IoError {
                message: format!("Failed to write padding: {e}"),
            })?;
        }

        // Tensor index
        file.write_all(&tensor_index_bytes).map_err(|e| RealizarError::IoError {
            message: format!("Failed to write tensor index: {e}"),
        })?;

        // Alignment padding before data
        let pre_data_padding = (data_offset_aligned - data_offset) as usize;
        if pre_data_padding > 0 {
            file.write_all(&vec![0u8; pre_data_padding]).map_err(|e| RealizarError::IoError {
                message: format!("Failed to write data alignment: {e}"),
            })?;
        }

        // Tensor data (with alignment)
        for tensor in &raw_tensors {
            file.write_all(&tensor.data).map_err(|e| RealizarError::IoError {
                message: format!("Failed to write tensor '{}': {e}", tensor.name),
            })?;

            // Align to 64 bytes
            let pad = (ALIGNMENT - (tensor.data.len() % ALIGNMENT)) % ALIGNMENT;
            if pad > 0 {
                file.write_all(&vec![0u8; pad]).map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write tensor padding: {e}"),
                })?;
            }
        }

        Ok(Q4KConversionStats {
            tensor_count: raw_tensors.len(),
            q4k_tensor_count: q4k_count,
            total_bytes,
            architecture: architecture.to_string(),
            num_layers: num_layers as usize,
            hidden_size: hidden_size as usize,
        })
    }
}

/// Statistics from Q4K conversion
#[derive(Debug, Clone)]
pub struct Q4KConversionStats {
    /// Total number of tensors
    pub tensor_count: usize,
    /// Number of Q4K quantized tensors
    pub q4k_tensor_count: usize,
    /// Total bytes written
    pub total_bytes: usize,
    /// Model architecture
    pub architecture: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden size
    pub hidden_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // Converter Tests
    // ==========================================================================

    #[test]
    fn test_from_gguf_transformer_config_preserved() {
        // Create a mock GGUF transformer
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.architecture, gguf.config.architecture);
        assert_eq!(apr.config.hidden_dim, gguf.config.hidden_dim);
        assert_eq!(apr.config.num_layers, gguf.config.num_layers);
        assert_eq!(apr.config.vocab_size, gguf.config.vocab_size);
    }

    #[test]
    fn test_from_gguf_transformer_weights_preserved() {
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.token_embedding, gguf.token_embedding);
        assert_eq!(apr.output_norm_weight, gguf.output_norm_weight);
        assert_eq!(apr.lm_head_weight, gguf.lm_head_weight);
    }

    #[test]
    fn test_from_gguf_transformer_layers_preserved() {
        let gguf = create_mock_gguf_transformer(4, 2, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.layers.len(), gguf.layers.len());
        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_weight, gguf_layer.attn_norm_weight);
            assert_eq!(apr_layer.qkv_weight, gguf_layer.qkv_weight);
            assert_eq!(apr_layer.ffn_up_weight, gguf_layer.ffn_up_weight);
            assert_eq!(apr_layer.ffn_down_weight, gguf_layer.ffn_down_weight);
        }
    }

    // ==========================================================================
    // APR Serialization Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_header_valid() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Check header (APR v2 format)
        assert_eq!(&bytes[0..4], &MAGIC); // APR2 magic
        assert_eq!(bytes[4], 2); // version major (v2)
        assert_eq!(bytes[5], 0); // version minor

        // Check tensor count (at bytes 8-11 in v2)
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(tensor_count, 1); // We store weights as single tensor
    }

    #[test]
    fn test_apr_bytes_roundtrip() {
        let original = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config, loaded.config);
        assert_eq!(original.token_embedding, loaded.token_embedding);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    #[test]
    fn test_from_apr_bytes_missing_weights() {
        // Create bytes with valid v2 header but no weights tensor
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&66u64.to_le_bytes()); // data offset (same = empty index)
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: no weights tensor
    }

    // ==========================================================================
    // Stats Tests
    // ==========================================================================

    #[test]
    fn test_stats_basic() {
        let apr = create_test_apr_transformer(64, 2, 1000, 256);
        let stats = GgufToAprConverter::stats(&apr);

        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.hidden_dim, 64);
        assert_eq!(stats.vocab_size, 1000);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
    }

    #[test]
    fn test_stats_memory_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        // Memory should be params * 4 bytes
        assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);

        // MB should be bytes / 1M
        let expected_mb = stats.memory_bytes_f32 as f64 / (1024.0 * 1024.0);
        assert!((stats.memory_mb() - expected_mb).abs() < 0.0001);
    }

    #[test]
    fn test_stats_parameter_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        let expected_m = stats.total_parameters as f64 / 1_000_000.0;
        assert!((stats.parameters_m() - expected_m).abs() < 0.0001);
    }

    // ==========================================================================
    // Inference Equivalence Tests
    // ==========================================================================

    #[test]
    fn test_inference_produces_output() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let result = apr.forward(&tokens);
        assert!(result.is_ok());

        let logits = result.expect("forward");
        assert_eq!(logits.len(), apr.config.vocab_size);
    }

    #[test]
    fn test_inference_deterministic() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let logits1 = apr.forward(&tokens).expect("forward 1");
        let logits2 = apr.forward(&tokens).expect("forward 2");

        assert_eq!(logits1, logits2, "Inference should be deterministic");
    }

    // ==========================================================================
    // Helper Functions
    // ==========================================================================

    fn create_mock_gguf_transformer(
        hidden_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        intermediate_dim: usize,
    ) -> GGUFTransformer {
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

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
            rope_type: 0, // NORM style (adjacent pairs)
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

    fn create_test_apr_transformer(
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

    // ==========================================================================
    // ConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_stats_memory_gb() {
        let stats = ConversionStats {
            total_parameters: 1_000_000_000, // 1B params
            memory_bytes_f32: 4_000_000_000, // 4GB
            num_layers: 24,
            hidden_dim: 2048,
            vocab_size: 50000,
            architecture: "test".to_string(),
        };

        let expected_gb = 4.0 / 1.073741824; // 4GB / GiB conversion
        assert!((stats.memory_gb() - expected_gb).abs() < 0.1);
    }

    #[test]
    fn test_stats_parameters_b() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000, // 7B params
            memory_bytes_f32: 28_000_000_000, // 28GB
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };

        assert!((stats.parameters_b() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 32,
            vocab_size: 100,
            architecture: "mini".to_string(),
        };

        // Test Debug trait
        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("mini"));
        assert!(debug_str.contains("1000"));
    }

    #[test]
    fn test_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 500,
            memory_bytes_f32: 2000,
            num_layers: 2,
            hidden_dim: 16,
            vocab_size: 50,
            architecture: "tiny".to_string(),
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    // ==========================================================================
    // Error Path Coverage Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_tensor_index() {
        // Create bytes with valid v2 header but truncated before tensor index
        let mut bytes = vec![0u8; 80]; // Just past header
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset beyond end
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated
    }

    #[test]
    fn test_from_apr_bytes_truncated_tensor_data() {
        // Create bytes with valid header and index but truncated data
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&110u64.to_le_bytes()); // data starts at 110
        bytes[64..66].copy_from_slice(b"{}");

        // Add a tensor index entry pointing to data beyond file end
        let index_json = r#"[{"name":"weights","dtype":"json","shape":[1000],"offset":0,"size":1000}]"#;
        let index_bytes = index_json.as_bytes();
        let index_end = 66 + index_bytes.len();
        bytes.resize(index_end + 10, 0); // Only add 10 bytes, not 1000
        bytes[66..index_end].copy_from_slice(index_bytes);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated tensor data
    }

    #[test]
    fn test_from_apr_bytes_invalid_json_tensor_index() {
        // Create bytes with valid header but invalid JSON in tensor index
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // index at 66
        bytes[32..40].copy_from_slice(&90u64.to_le_bytes()); // data at 90
        bytes[64..66].copy_from_slice(b"{}");
        // Invalid JSON at tensor index position
        bytes[66..78].copy_from_slice(b"not valid js");

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: invalid JSON
    }

    // ==========================================================================
    // RawTensor Coverage Tests
    // ==========================================================================

    #[test]
    fn test_raw_tensor_debug() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![0u8; 100],
            shape: vec![10, 10],
            dtype: 0, // F32
        };

        let debug_str = format!("{tensor:?}");
        assert!(debug_str.contains("test.weight"));
        assert!(debug_str.contains("[10, 10]"));
    }

    #[test]
    fn test_raw_tensor_clone() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 1, // F16
        };

        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }

    // ==========================================================================
    // GgufToAprQ4KConverter Helper Tests
    // ==========================================================================

    #[test]
    fn test_get_string_helper() {
        use std::collections::HashMap;
        use crate::gguf::GGUFValue;

        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), GGUFValue::String("test_model".to_string()));
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "name");
        assert_eq!(result, Some("test_model".to_string()));

        let missing = GgufToAprQ4KConverter::get_string(&metadata, "nonexistent");
        assert_eq!(missing, None);

        // Test wrong type returns None
        let wrong_type = GgufToAprQ4KConverter::get_string(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_u32_helper() {
        use std::collections::HashMap;
        use crate::gguf::GGUFValue;

        let mut metadata = HashMap::new();
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));
        metadata.insert("signed".to_string(), GGUFValue::Int32(100));
        metadata.insert("big".to_string(), GGUFValue::UInt64(200));
        metadata.insert("name".to_string(), GGUFValue::String("test".to_string()));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "count");
        assert_eq!(result, Some(42));

        let signed = GgufToAprQ4KConverter::get_u32(&metadata, "signed");
        assert_eq!(signed, Some(100));

        let big = GgufToAprQ4KConverter::get_u32(&metadata, "big");
        assert_eq!(big, Some(200));

        let missing = GgufToAprQ4KConverter::get_u32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_u32(&metadata, "name");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_f32_helper() {
        use std::collections::HashMap;
        use crate::gguf::GGUFValue;

        let mut metadata = HashMap::new();
        metadata.insert("scale".to_string(), GGUFValue::Float32(3.14));
        metadata.insert("big_scale".to_string(), GGUFValue::Float64(2.71828));
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "scale");
        assert!(result.is_some());
        assert!((result.unwrap() - 3.14).abs() < 0.001);

        let big = GgufToAprQ4KConverter::get_f32(&metadata, "big_scale");
        assert!(big.is_some());
        assert!((big.unwrap() - 2.71828).abs() < 0.001);

        let missing = GgufToAprQ4KConverter::get_f32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_f32(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    // ==========================================================================
    // Q4KConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_q4k_conversion_stats_debug() {
        let stats = Q4KConversionStats {
            tensor_count: 100,
            q4k_tensor_count: 80,
            total_bytes: 1_000_000,
            architecture: "llama".to_string(),
            num_layers: 32,
            hidden_size: 4096,
        };

        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("llama"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("32"));
    }

    #[test]
    fn test_q4k_conversion_stats_clone() {
        let stats = Q4KConversionStats {
            tensor_count: 50,
            q4k_tensor_count: 40,
            total_bytes: 500_000,
            architecture: "qwen".to_string(),
            num_layers: 16,
            hidden_size: 2048,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.tensor_count, stats.tensor_count);
        assert_eq!(cloned.architecture, stats.architecture);
        assert_eq!(cloned.num_layers, stats.num_layers);
    }

    // ==========================================================================
    // Additional From APR Bytes Error Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_v1_format() {
        // Create APR v1 format header (should be handled or error gracefully)
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 1; // v1 (not v2)
        bytes[5] = 0;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // May succeed with v1 fallback or fail, but shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_from_apr_bytes_wrong_magic() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic
        bytes[4] = 2;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_too_short() {
        // Only 4 bytes (magic only)
        let bytes = vec![0x41, 0x50, 0x52, 0x32]; // APR2

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Stats Edge Cases
    // ==========================================================================

    #[test]
    fn test_stats_zero_params() {
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
    fn test_stats_small_model() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 16,
            vocab_size: 100,
            architecture: "tiny".to_string(),
        };

        assert!(stats.memory_mb() > 0.0);
        assert!(stats.parameters_m() > 0.0);
        assert!(stats.parameters_b() < 0.001);
    }

    // ==========================================================================
    // APR Bytes Serialization Additional Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_multiple_layers() {
        let apr = create_test_apr_transformer(64, 4, 1000, 256);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should have valid header
        assert_eq!(&bytes[0..4], &MAGIC);
        assert!(bytes.len() > HEADER_SIZE);
    }

    #[test]
    fn test_to_apr_bytes_single_layer() {
        let apr = create_test_apr_transformer(32, 1, 100, 64);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert_eq!(&bytes[0..4], &MAGIC);
    }

    #[test]
    fn test_apr_roundtrip_multiple_layers() {
        let original = create_test_apr_transformer(32, 3, 500, 128);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config.num_layers, loaded.config.num_layers);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    // ==========================================================================
    // Coverage Tests: ConversionStats
    // ==========================================================================

    #[test]
    fn test_conversion_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50000,
            architecture: "bert".to_string(),
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ConversionStats"));
        assert!(debug_str.contains("1000000"));
        assert!(debug_str.contains("bert"));
    }

    #[test]
    fn test_conversion_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    #[test]
    fn test_conversion_stats_large_model() {
        let stats = ConversionStats {
            total_parameters: 70_000_000_000, // 70B
            memory_bytes_f32: 280_000_000_000,
            num_layers: 80,
            hidden_dim: 8192,
            vocab_size: 128000,
            architecture: "llama3".to_string(),
        };
        assert!(stats.parameters_b() > 69.0 && stats.parameters_b() < 71.0);
        assert!(stats.memory_gb() > 250.0);
    }

    // ==========================================================================
    // Coverage Tests: from_apr_bytes error paths
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_invalid_tensor_index_json() {
        // Create header pointing to invalid JSON
        let mut bytes = vec![0u8; 200];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&100u64.to_le_bytes()); // data offset
        bytes[64..66].copy_from_slice(b"{}"); // metadata
        // Invalid JSON for tensor index (length must match exactly)
        let invalid_json = b"not valid json{{{";
        bytes[66..66 + invalid_json.len()].copy_from_slice(invalid_json);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Coverage Tests: GgufToAprQ4KConverter helpers
    // ==========================================================================

    #[test]
    fn test_q4k_converter_get_string_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_string(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_string_present() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_int32() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(42));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(100));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_q4k_converter_get_f32_from_float64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(3.14159));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.unwrap() - 3.14159).abs() < 0.0001);
    }

    #[test]
    fn test_q4k_converter_get_string_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("not a number".to_string()));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("not a float".to_string()));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_none());
    }

}
