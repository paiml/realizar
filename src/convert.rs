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
}
