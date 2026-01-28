//! Quantized GGUF transformer types
//!
//! This module contains the quantized transformer layer and model structures
//! that enable fused dequantization operations for memory-efficient inference.

use crate::error::{RealizarError, Result};
use crate::quantize::QK_K;

use super::config::GGUFConfig;
use super::quantized::{QKVWeights, QuantizedTensorRef};
use super::types::{
    GGUFModel, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};

/// Quantized transformer layer weights (stored as byte references)
///
/// Unlike `GGUFTransformerLayer` which stores dequantized Vec<f32>,
/// this stores references to quantized data for fused operations.
pub struct QuantizedGGUFTransformerLayer {
    /// Attention norm weight (kept as f32 - small, read once per token)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (quantized) - supports fused or separate
    pub qkv_weight: QKVWeights,
    /// QKV bias (optional, f32)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_output_weight: QuantizedTensorRef,
    /// Attention output bias (optional, f32)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN up projection (quantized)
    pub ffn_up_weight: QuantizedTensorRef,
    /// FFN up bias (optional, f32)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection (quantized)
    pub ffn_down_weight: QuantizedTensorRef,
    /// FFN down bias (optional, f32)
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN gate projection (quantized, SwiGLU models like LLaMA)
    pub ffn_gate_weight: Option<QuantizedTensorRef>,
    /// FFN gate bias (optional, f32)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN norm weight (pre-FFN layer norm, LLaMA-style)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional, f32)
    pub ffn_norm_bias: Option<Vec<f32>>,
}

/// Quantized GGUF Transformer for fused inference
///
/// Per Williams et al. (2009) roofline model, LLM inference is memory-bound.
/// This transformer stores weights in quantized form and uses fused
/// dequant+dot operations to minimize memory bandwidth.
///
/// # Performance Benefits
///
/// - **8x bandwidth reduction** for Q4_K vs f32 (144 bytes vs 1024 bytes per 256 values)
/// - **Zero intermediate buffers** - dequantization happens inline with dot product
/// - **SIMD acceleration** - AVX2/FMA fused operations when available
/// - **Zero-copy loading** - weights stay in memory-mapped file
///
/// # Architecture
///
/// ```text
/// [Memory-mapped Q4_K bytes] → [fused_q4k_dot_simd] → [f32 result]
///                               ↑
///                         No intermediate Vec<f32>!
/// ```
pub struct QuantizedGGUFTransformer<'a> {
    /// Model configuration
    pub config: GGUFConfig,
    /// Reference to memory-mapped file data
    pub data: &'a [u8],
    /// Token embedding (kept as f32 for lookup)
    pub token_embedding: Vec<f32>,
    /// Quantized layer weights
    pub layers: Vec<QuantizedGGUFTransformerLayer>,
    /// Output norm weight (f32)
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (quantized for large vocab)
    pub lm_head_weight: QuantizedTensorRef,
    /// LM head bias (optional, f32)
    pub lm_head_bias: Option<Vec<f32>>,
}

impl<'a> QuantizedGGUFTransformer<'a> {
    /// Load quantized transformer from memory-mapped GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model metadata
    /// * `data` - Memory-mapped file data (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or have unsupported format
    pub fn from_gguf(model: &GGUFModel, data: &'a [u8]) -> Result<Self> {
        let config = GGUFConfig::from_gguf(model)?;

        // Token embedding - keep as f32 for efficient lookup
        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;

        // Load layers with quantized weight references
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_quantized_layer(model, data, layer_idx)?;
            layers.push(layer);
        }

        // Output norm - small, keep as f32
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;
        let output_norm_bias = model.get_tensor_f32("output_norm.bias", data).ok();

        // LM head - large, keep quantized
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")
            .or_else(|_| Self::get_tensor_ref(model, data, "token_embd.weight"))?;
        let lm_head_bias = model.get_tensor_f32("output.bias", data).ok();

        Ok(Self {
            config,
            data,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// Get tensor reference (offset + size + qtype) without dequantization
    fn get_tensor_ref(model: &GGUFModel, data: &[u8], name: &str) -> Result<QuantizedTensorRef> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor '{}' not found", name),
            })?;

        let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
        let offset = model.tensor_data_start + tensor.offset as usize;

        // Calculate byte size based on quantization type
        let byte_size = match tensor.qtype {
            GGUF_TYPE_F32 => num_elements * 4,
            GGUF_TYPE_Q4_0 => {
                // Q4_0: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 16 bytes (32×4-bit values) = 18 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 18;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q8_0 => {
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q2_K => {
                // Q2_K: 256 elements per super-block
                // Layout: 16 bytes scales + 64 bytes quants + 2 bytes d + 2 bytes dmin = 84 bytes
                const SUPER_BLOCK_BYTES: usize = 84;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q4_1 => {
                // Q4_1: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 1×f16 min (2 bytes) + 16 bytes (32×4-bit values) = 20 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 20;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q5_0 => {
                // Q5_0: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 4 bytes high bits + 16 bytes quants = 22 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 22;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q4_K => {
                const SUPER_BLOCK_BYTES: usize = 144;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q5_K => {
                const SUPER_BLOCK_BYTES: usize = 176;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q6_K => {
                const SUPER_BLOCK_BYTES: usize = 210;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            _ => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "get_tensor_ref".to_string(),
                    reason: format!("Unsupported quantization type: {}", tensor.qtype),
                });
            },
        };

        // PAR-058-RESOLVED: Validate byte size and auto-correct qtype if mismatch detected
        // Some GGUF files have incorrect qtype in header (e.g., Q5_0 header but Q4_0 data)
        // Detect this by checking if the calculated byte_size would exceed file bounds,
        // and try alternative qtypes that match the actual data size.
        let (byte_size, actual_qtype) = {
            // Try the claimed qtype first
            if offset + byte_size <= data.len() {
                (byte_size, tensor.qtype)
            } else {
                // Mismatch! Try to infer correct qtype from available data
                // This happens when GGUF header has wrong qtype (e.g., qwen2.5-coder-0.5b)
                let avail = data.len().saturating_sub(offset);

                // Try Q4_0 (18 bytes per 32 elements)
                let q4_0_size = {
                    const BLOCK_SIZE: usize = 32;
                    const BLOCK_BYTES: usize = 18;
                    num_elements.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
                };
                if q4_0_size <= avail && q4_0_size > 0 {
                    eprintln!(
                        "[PAR-058-RESOLVED] Tensor '{}' qtype mismatch: header says {} but byte size suggests Q4_0. Using Q4_0.",
                        name, tensor.qtype
                    );
                    (q4_0_size, GGUF_TYPE_Q4_0)
                } else {
                    // Try Q8_0 (34 bytes per 32 elements)
                    let q8_0_size = {
                        const BLOCK_SIZE: usize = 32;
                        const BLOCK_BYTES: usize = 34;
                        num_elements.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
                    };
                    if q8_0_size <= avail && q8_0_size > 0 {
                        eprintln!(
                            "[PAR-058-RESOLVED] Tensor '{}' qtype mismatch: header says {} but byte size suggests Q8_0. Using Q8_0.",
                            name, tensor.qtype
                        );
                        (q8_0_size, GGUF_TYPE_Q8_0)
                    } else {
                        // Fallback to original (will fail bounds check below)
                        (byte_size, tensor.qtype)
                    }
                }
            }
        };

        // Validate bounds
        if offset + byte_size > data.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Tensor '{}' data range [{}, {}) exceeds file size {}",
                    name,
                    offset,
                    offset + byte_size,
                    data.len()
                ),
            });
        }

        Ok(QuantizedTensorRef {
            offset,
            byte_size,
            num_elements,
            qtype: actual_qtype, // PAR-058-RESOLVED: Use auto-corrected qtype
        })
    }

    /// Load a single quantized transformer layer
    fn load_quantized_layer(
        model: &GGUFModel,
        data: &[u8],
        layer_idx: usize,
    ) -> Result<QuantizedGGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm - small, keep as f32
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
            .ok();

        // QKV - large, keep quantized
        // Try fused first (phi-2 style), fall back to separate (llama style)
        let (qkv_weight, qkv_bias) = if let Ok(fused) =
            Self::get_tensor_ref(model, data, &format!("{}.attn_qkv.weight", prefix))
        {
            // phi-2 style: fused QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
                .ok();
            (QKVWeights::Fused(fused), bias)
        } else {
            // llama style: separate Q, K, V tensors
            let q = Self::get_tensor_ref(model, data, &format!("{}.attn_q.weight", prefix))?;
            let k = Self::get_tensor_ref(model, data, &format!("{}.attn_k.weight", prefix))?;
            let v = Self::get_tensor_ref(model, data, &format!("{}.attn_v.weight", prefix))?;

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let mut combined = Vec::with_capacity(qb.len() + kb.len() + vb.len());
                    combined.extend_from_slice(&qb);
                    combined.extend_from_slice(&kb);
                    combined.extend_from_slice(&vb);
                    Some(combined)
                },
                _ => None,
            };

            (QKVWeights::Separate { q, k, v }, bias)
        };

        // Attention output - large, keep quantized
        let attn_output_weight =
            Self::get_tensor_ref(model, data, &format!("{}.attn_output.weight", prefix))?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
            .ok();

        // FFN - large, keep quantized
        let ffn_up_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_up.weight", prefix))?;
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
            .ok();
        let ffn_down_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_down.weight", prefix))?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
            .ok();

        // FFN gate - SwiGLU models like LLaMA have this
        let ffn_gate_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_gate.weight", prefix)).ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), data)
            .ok();

        // FFN norm - LLaMA-style pre-FFN layer norm
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
            .ok();
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
            .ok();

        Ok(QuantizedGGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_norm_weight,
            ffn_norm_bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};

    // ============================================================================
    // QuantizedGGUFTransformerLayer tests
    // ============================================================================

    #[test]
    fn test_quantized_layer_struct_fields() {
        // Verify struct field layout and optional fields
        let layer = QuantizedGGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: QKVWeights::Fused(QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 192,
                qtype: GGUF_TYPE_Q4_K,
            }),
            qkv_bias: None,
            attn_output_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 50,
                num_elements: 64,
                qtype: GGUF_TYPE_Q4_K,
            },
            attn_output_bias: None,
            ffn_up_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 256,
                qtype: GGUF_TYPE_Q4_K,
            },
            ffn_up_bias: None,
            ffn_down_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 64,
                qtype: GGUF_TYPE_Q4_K,
            },
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };

        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert!(layer.attn_norm_bias.is_none());
        assert!(layer.ffn_gate_weight.is_none());
    }

    #[test]
    fn test_quantized_layer_with_optional_fields() {
        let layer = QuantizedGGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 32],
            attn_norm_bias: Some(vec![0.0; 32]),
            qkv_weight: QKVWeights::Separate {
                q: QuantizedTensorRef { offset: 0, byte_size: 50, num_elements: 32, qtype: GGUF_TYPE_Q4_0 },
                k: QuantizedTensorRef { offset: 50, byte_size: 50, num_elements: 32, qtype: GGUF_TYPE_Q4_0 },
                v: QuantizedTensorRef { offset: 100, byte_size: 50, num_elements: 32, qtype: GGUF_TYPE_Q4_0 },
            },
            qkv_bias: Some(vec![0.0; 96]),
            attn_output_weight: QuantizedTensorRef { offset: 0, byte_size: 50, num_elements: 32, qtype: GGUF_TYPE_Q4_0 },
            attn_output_bias: Some(vec![0.0; 32]),
            ffn_up_weight: QuantizedTensorRef { offset: 0, byte_size: 100, num_elements: 128, qtype: GGUF_TYPE_Q4_0 },
            ffn_up_bias: Some(vec![0.0; 128]),
            ffn_down_weight: QuantizedTensorRef { offset: 0, byte_size: 50, num_elements: 32, qtype: GGUF_TYPE_Q4_0 },
            ffn_down_bias: Some(vec![0.0; 32]),
            ffn_gate_weight: Some(QuantizedTensorRef { offset: 0, byte_size: 100, num_elements: 128, qtype: GGUF_TYPE_Q4_0 }),
            ffn_gate_bias: Some(vec![0.0; 128]),
            ffn_norm_weight: Some(vec![1.0; 32]),
            ffn_norm_bias: Some(vec![0.0; 32]),
        };

        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    // ============================================================================
    // QuantizedGGUFTransformer tests using test_factory
    // ============================================================================

    #[test]
    fn test_quantized_transformer_from_llama_gguf() {
        // build_minimal_llama_gguf(vocab_size, hidden_dim, intermediate_dim, num_heads, num_kv_heads)
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data);

        assert!(transformer.is_ok(), "Failed to load transformer: {:?}", transformer.err());
        let transformer = transformer.unwrap();

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.config.num_layers, 1); // test_factory builds 1 layer
        assert_eq!(transformer.layers.len(), 1);
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
    }

    #[test]
    fn test_quantized_transformer_from_phi2_gguf() {
        // build_minimal_phi2_gguf(vocab_size, hidden_dim, intermediate_dim, num_heads)
        let gguf_data = build_minimal_phi2_gguf(100, 64, 256, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data);

        assert!(transformer.is_ok(), "Failed to load transformer: {:?}", transformer.err());
        let transformer = transformer.unwrap();

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.layers.len(), 1);
    }

    #[test]
    fn test_quantized_transformer_config() {
        let gguf_data = build_minimal_llama_gguf(200, 128, 512, 8, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        assert_eq!(transformer.config.hidden_dim, 128);
        // intermediate_dim is inferred from ffn_up tensor, may differ from input
        assert!(transformer.config.intermediate_dim > 0);
        assert_eq!(transformer.config.num_heads, 8);
        assert_eq!(transformer.config.num_kv_heads, 4);
        assert_eq!(transformer.config.vocab_size, 200);
    }

    #[test]
    fn test_quantized_transformer_output_norm() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        // Output norm weight should be hidden_dim size
        assert_eq!(transformer.output_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_transformer_lm_head() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        // LM head should have elements (may use tied embeddings with F32 type 0)
        assert!(transformer.lm_head_weight.num_elements > 0);
        // qtype can be 0 (F32) for tied embeddings
        assert!(transformer.lm_head_weight.byte_size > 0);
    }

    #[test]
    fn test_quantized_transformer_layer_attn_norm() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_transformer_layer_qkv() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        let layer = &transformer.layers[0];
        // QKV should be either fused or separate
        match &layer.qkv_weight {
            QKVWeights::Fused(ref tensor) => {
                assert!(tensor.num_elements > 0);
            }
            QKVWeights::Separate { q, k, v } => {
                assert!(q.num_elements > 0);
                assert!(k.num_elements > 0);
                assert!(v.num_elements > 0);
            }
        }
    }

    #[test]
    fn test_quantized_transformer_layer_ffn() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        let layer = &transformer.layers[0];
        assert!(layer.ffn_up_weight.num_elements > 0);
        assert!(layer.ffn_down_weight.num_elements > 0);
    }

    #[test]
    fn test_quantized_transformer_has_data_ref() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).unwrap();

        // The data reference should point to the original data
        assert!(!transformer.data.is_empty());
        assert_eq!(transformer.data.len(), gguf_data.len());
    }
}
