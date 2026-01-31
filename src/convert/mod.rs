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
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
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
        header[40..44].copy_from_slice(&0u32.to_le_bytes()); // checksum: reserved for future use
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
    fn get_string(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<String> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Helper to extract u32 from GGUF metadata
    fn get_u32(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<u32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::UInt32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Int32(v)) => Some(*v as u32),
            Some(crate::gguf::GGUFValue::UInt64(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// Helper to extract f32 from GGUF metadata
    fn get_f32(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<f32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::Float32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// PMAT-107: Infer rope_type from architecture (matches llama.cpp llama-model.cpp:7763-7811)
    ///
    /// Returns:
    /// - 0 = NORM style (adjacent pairs) - default for LLaMA, TinyLlama
    /// - 2 = NEOX style (split halves) - for Qwen2, Phi3, Gemma, etc.
    fn infer_rope_type(
        architecture: &str,
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
    ) -> u32 {
        // First check for explicit rope.scaling.type in metadata
        let scaling_key = format!("{}.rope.scaling.type", architecture);
        if let Some(crate::gguf::GGUFValue::String(s)) = metadata.get(&scaling_key) {
            match s.as_str() {
                "none" | "linear" => return 0, // NORM style
                "yarn" | "neox" => return 2,   // NEOX style
                _ => {}
            }
        }

        // Infer from architecture name (matches llama.cpp neox-style architectures)
        let arch_lower = architecture.to_lowercase();
        let neox_architectures = [
            "qwen",
            "qwen2",
            "qwen3",
            "stablelm",
            "phi2",
            "phi3",
            "gemma",
            "gemma2",
            "gemma3",
            "starcoder2",
            "gptneox",
            "falcon",
            "codeshell",
            "orion",
            "bert",
            "nomic-bert",
            "dbrx",
            "olmo2",
            "olmoe",
            "plamo",
            "plamo2",
            "openelm",
            "exaone",
            "minicpm3",
            "nemotron",
            "internlm2",
            "deepseek2",
        ];

        for neox_arch in neox_architectures {
            if arch_lower.contains(neox_arch) {
                return 2; // NEOX style
            }
        }

        // Default to NORM style (LLaMA, TinyLlama, etc.)
        0
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
    pub fn convert(
        gguf_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<Q4KConversionStats> {
        use std::io::Write;

        // Load GGUF with raw quantized tensors
        let gguf_data = std::fs::read(gguf_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read GGUF: {e}"),
        })?;

        let gguf_model = crate::gguf::GGUFModel::from_bytes(&gguf_data)?;

        // Extract model config from metadata
        let architecture = Self::get_string(&gguf_model.metadata, "general.architecture")
            .unwrap_or_else(|| "unknown".to_string());
        let hidden_size = Self::get_u32(
            &gguf_model.metadata,
            &format!("{architecture}.embedding_length"),
        )
        .unwrap_or(0);
        let num_layers =
            Self::get_u32(&gguf_model.metadata, &format!("{architecture}.block_count"))
                .unwrap_or(0);
        let num_heads = Self::get_u32(
            &gguf_model.metadata,
            &format!("{architecture}.attention.head_count"),
        )
        .unwrap_or(0);
        let num_kv_heads = Self::get_u32(
            &gguf_model.metadata,
            &format!("{architecture}.attention.head_count_kv"),
        )
        .unwrap_or(num_heads);
        let vocab_size = Self::get_u32(&gguf_model.metadata, &format!("{architecture}.vocab_size"))
            .or_else(|| Self::get_u32(&gguf_model.metadata, "tokenizer.ggml.vocab_size"))
            .unwrap_or_else(|| {
                // Infer from embedding tensor shape if metadata not available
                gguf_model
                    .tensors
                    .iter()
                    .find(|t| {
                        t.name.contains("token_embd")
                            || t.name.contains("embed_tokens")
                            || t.name.contains("tok_embeddings")
                    })
                    .and_then(|t| t.dims.first().copied().map(|d| d as u32))
                    .unwrap_or(0)
            }) as usize;
        let intermediate_size = Self::get_u32(
            &gguf_model.metadata,
            &format!("{architecture}.feed_forward_length"),
        )
        .unwrap_or(0);
        let context_length = Self::get_u32(
            &gguf_model.metadata,
            &format!("{architecture}.context_length"),
        )
        .unwrap_or(2048);
        let rope_theta = Self::get_f32(
            &gguf_model.metadata,
            &format!("{architecture}.rope.freq_base"),
        )
        .unwrap_or(10000.0);
        let eps = Self::get_f32(
            &gguf_model.metadata,
            &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        )
        .unwrap_or(1e-5);

        // PMAT-107: Infer rope_type from architecture (matches llama.cpp llama-model.cpp:7763-7811)
        // NEOX style (type 2) uses split-halves, NORM style (type 0) uses adjacent pairs
        let rope_type = Self::infer_rope_type(&architecture, &gguf_model.metadata);

        // Build metadata JSON
        // F-REGR-231 FIX: Use field names consistent with AprTransformer::from_apr_bytes
        let metadata = serde_json::json!({
            "model_type": "transformer_lm_q4k",
            "architecture": architecture,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,  // Loader checks num_hidden_layers first
            "num_attention_heads": num_heads, // Loader checks num_attention_heads first
            "num_key_value_heads": num_kv_heads, // Loader checks num_key_value_heads first
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size, // Loader checks intermediate_size first
            "max_position_embeddings": context_length, // Loader checks max_position_embeddings
            "rope_theta": rope_theta,
            "rope_type": rope_type,
            "rms_norm_eps": eps,  // F-REGR-231: Was "eps", loader reads "rms_norm_eps"
            "quantization": "Q4_K_M",
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
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
                0 => num_elements * 4,            // F32
                1 => num_elements * 2,            // F16
                8 => (num_elements / 32) * 34,    // Q8_0: 32 elements = 2 (scale) + 32 (quants)
                12 => (num_elements / 256) * 144, // Q4_K: 256 elements = 144 bytes
                13 => (num_elements / 256) * 176, // Q5_K: 256 elements = 176 bytes
                14 => (num_elements / 256) * 210, // Q6_K: 256 elements = 210 bytes
                _ => num_elements * 4,            // Default to F32
            };

            // Extract raw bytes
            let tensor_start = gguf_model.tensor_data_start + tensor_meta.offset as usize;
            if tensor_start + byte_size > gguf_data.len() {
                return Err(RealizarError::FormatError {
                    reason: format!(
                        "Tensor '{}' exceeds file bounds (start={}, size={}, file_len={})",
                        name,
                        tensor_start,
                        byte_size,
                        gguf_data.len()
                    ),
                });
            }

            let data = gguf_data[tensor_start..tensor_start + byte_size].to_vec();

            // Q4_K is GGML type 12
            if qtype == 12 {
                q4k_count += 1;
            }
            total_bytes += byte_size;

            raw_tensors.push(RawTensor {
                name,
                data,
                shape,
                dtype: qtype,
            });
        }

        // Build binary tensor index
        let mut tensor_index_bytes: Vec<u8> = Vec::new();
        let mut current_offset = 0u64;

        for tensor in &raw_tensors {
            // name_len (2 bytes) + name
            let name_bytes = tensor.name.as_bytes();
            tensor_index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            tensor_index_bytes.extend_from_slice(name_bytes);

            // dtype (1 byte) - write GGML dtype directly
            // The APR v2 TensorEntry::from_binary reader handles these values:
            //   0=F32, 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
            // GH-191 FIX: Previously used wrong APR-specific dtype codes (8=Q4_K, 9=Q6_K, 10=Q8_0)
            // that didn't match the reader's mapping, causing all tensors to load as F32.
            let apr_dtype = match tensor.dtype {
                0 => 0u8,   // F32
                1 => 1u8,   // F16
                8 => 8u8,   // Q8_0 (GGML type 8)
                12 => 12u8, // Q4_K (GGML type 12)
                13 => 13u8, // Q5_K (GGML type 13)
                14 => 14u8, // Q6_K (GGML type 14)
                other => {
                    eprintln!("WARN: Unknown GGML dtype {other} for tensor '{}', writing as F32", tensor.name);
                    0u8
                }
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
        file.write_all(&header)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write header: {e}"),
            })?;

        // Metadata (padded)
        file.write_all(&metadata_bytes)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write metadata: {e}"),
            })?;
        let padding = metadata_padded_len - metadata_bytes.len();
        if padding > 0 {
            file.write_all(&vec![0u8; padding])
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write padding: {e}"),
                })?;
        }

        // Tensor index
        file.write_all(&tensor_index_bytes)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write tensor index: {e}"),
            })?;

        // Alignment padding before data
        let pre_data_padding = (data_offset_aligned - data_offset) as usize;
        if pre_data_padding > 0 {
            file.write_all(&vec![0u8; pre_data_padding])
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write data alignment: {e}"),
                })?;
        }

        // Tensor data (with alignment)
        for tensor in &raw_tensors {
            file.write_all(&tensor.data)
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write tensor '{}': {e}", tensor.name),
                })?;

            // Align to 64 bytes
            let pad = (ALIGNMENT - (tensor.data.len() % ALIGNMENT)) % ALIGNMENT;
            if pad > 0 {
                file.write_all(&vec![0u8; pad])
                    .map_err(|e| RealizarError::IoError {
                        message: format!("Failed to write tensor padding: {e}"),
                    })?;
            }
        }

        Ok(Q4KConversionStats {
            tensor_count: raw_tensors.len(),
            q4k_tensor_count: q4k_count,
            total_bytes,
            architecture: architecture.clone(),
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

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod convert_tests;

// T-COV-95 Coverage Bridge tests (Part 02 - B5)
#[cfg(test)]
#[path = "tests_part_02.rs"]
mod convert_tests_part_02;

// T-COV-95 Deep Coverage Bridge (Part 03 - Q4K converter, rope_type, helpers)
#[cfg(test)]
#[path = "tests_part_03.rs"]
mod convert_tests_part_03;

// T-COV-95 Coverage Bridge (Part 04 - ConversionStats, to_apr_bytes, from_apr_bytes)
#[cfg(test)]
#[path = "tests_part_04.rs"]
mod convert_tests_part_04;

// T-COV-95 Extended Coverage (Part 05 - RawTensor, dtypes, edge cases)
#[cfg(test)]
#[path = "tests_part_05.rs"]
mod convert_tests_part_05;
