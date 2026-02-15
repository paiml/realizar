impl OwnedQuantizedModel {
    /// Create owned model from memory-mapped GGUF file
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    pub fn from_mapped(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        let data = mapped.data();
        let transformer = QuantizedGGUFTransformer::from_gguf(&mapped.model, data)?;

        // Get config for dimension calculations
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // Convert layers to owned (passing config for dimensions)
        let layers: Vec<OwnedQuantizedLayer> = transformer
            .layers
            .iter()
            .map(|l| OwnedQuantizedLayer::from_borrowed(l, data, config))
            .collect();

        Ok(Self {
            config: transformer.config.clone(),
            token_embedding: transformer.token_embedding,
            layers,
            output_norm_weight: transformer.output_norm_weight,
            output_norm_bias: transformer.output_norm_bias,
            // LM head: [hidden_dim] -> [vocab_size]
            lm_head_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &transformer.lm_head_weight,
                data,
                hidden_dim,
                vocab_size,
            ),
            lm_head_bias: transformer.lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Create a model for testing purposes
    ///
    /// This constructor handles the internal CUDA fields automatically,
    /// allowing external tests to construct models without accessing pub(crate) fields.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `token_embedding` - Token embedding weights
    /// * `layers` - Quantized transformer layers
    /// * `output_norm_weight` - Output normalization weight
    /// * `output_norm_bias` - Optional output normalization bias
    /// * `lm_head_weight` - Language model head weight
    /// * `lm_head_bias` - Optional language model head bias
    #[must_use]
    pub fn new_for_test(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<OwnedQuantizedLayer>,
        output_norm_weight: Vec<f32>,
        output_norm_bias: Option<Vec<f32>>,
        lm_head_weight: OwnedQuantizedTensor,
        lm_head_bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Create model from memory-mapped APR file (SHOWCASE-APR-GPU)
    ///
    /// Converts APR Q4K format to GGUF-compatible model for GPU inference.
    /// The raw Q4K tensor data is byte-compatible between formats.
    ///
    /// # Arguments
    /// * `apr` - Memory-mapped APR model
    ///
    /// # Errors
    /// Returns error if APR format is invalid or missing required tensors.
    pub fn from_apr(apr: &crate::apr::MappedAprModel) -> Result<Self> {
        use crate::apr::MappedAprModel;

        let data = apr.data();
        let data_offset = apr.data_offset() as usize;

        // Build config from APR metadata
        let hidden_dim = apr.metadata.hidden_size.unwrap_or(1536);
        let num_layers = apr.metadata.num_layers.unwrap_or(28);
        let num_heads = apr.metadata.num_heads.unwrap_or(12);
        let num_kv_heads = apr.metadata.num_kv_heads.unwrap_or(2);
        let intermediate_dim = apr.metadata.intermediate_size.unwrap_or(8960);
        let eps = apr.metadata.rms_norm_eps.unwrap_or(1e-6);
        let rope_theta = apr.metadata.rope_theta.unwrap_or(1_000_000.0);

        // Infer vocab_size from embedding tensor if metadata is 0 or missing
        let vocab_size = match apr.metadata.vocab_size {
            Some(v) if v > 0 => v,
            _ => {
                // Try to infer from embedding tensor shape
                apr.tensors
                    .iter()
                    .find(|t| {
                        t.name.contains("embed_tokens")
                            || t.name.contains("tok_embeddings")
                            || t.name.contains("token_embd")
                    })
                    .and_then(|t| t.shape.first().copied())
                    .unwrap_or(151936)
            },
        };

        let config = GGUFConfig {
            architecture: apr
                .metadata
                .architecture
                .clone()
                .unwrap_or_else(|| "qwen2".to_string()),
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_dim,
            eps,
            rope_theta,
            rope_type: 2, // NEOX style for Qwen2.5
            context_length: 32768,
            bos_token_id: apr.metadata.get_embedded_bos_token_id(),
        };

        // Helper to make OwnedQuantizedTensor (tries multiple names for GGUF/HF compat)
        let make_tensor =
            |names: &[&str], in_dim: usize, out_dim: usize| -> Result<OwnedQuantizedTensor> {
                let (tensor, found_name) = names
                    .iter()
                    .find_map(|name| apr.find_tensor(name).map(|t| (t, *name)))
                    .ok_or_else(|| RealizarError::FormatError {
                        reason: format!("APR: tensor not found (tried: {})", names.join(", ")),
                    })?;
                let start = data_offset + tensor.offset as usize;
                let end = start + tensor.size as usize;
                if end > data.len() {
                    return Err(RealizarError::FormatError {
                        reason: format!("APR: tensor {found_name} extends past EOF"),
                    });
                }
                let qtype = MappedAprModel::dtype_to_qtype(&tensor.dtype);
                Ok(OwnedQuantizedTensor {
                    data: data[start..end].to_vec(),
                    in_dim,
                    out_dim,
                    qtype,
                })
            };

        // Helper to get F32 tensor data (tries multiple names)
        let get_f32_tensor = |names: &[&str]| -> Result<Vec<f32>> {
            let (tensor, found_name) = names
                .iter()
                .find_map(|name| apr.find_tensor(name).map(|t| (t, *name)))
                .ok_or_else(|| RealizarError::FormatError {
                    reason: format!("APR: tensor not found (tried: {})", names.join(", ")),
                })?;
            let start = data_offset + tensor.offset as usize;
            let end = start + tensor.size as usize;
            if end > data.len() {
                return Err(RealizarError::FormatError {
                    reason: format!("APR: tensor {found_name} extends past EOF"),
                });
            }
            Ok(data[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        // Helper to try loading an optional F32 bias tensor
        let try_get_f32 = |name: &str| -> Option<Vec<f32>> {
            let tensor = apr.find_tensor(name)?;
            let start = data_offset + tensor.offset as usize;
            let end = start + tensor.size as usize;
            if end > data.len() {
                return None;
            }
            Some(
                data[start..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        };

        // Load token embeddings (F32)
        let embed_name = apr
            .tensors
            .iter()
            .find(|t| {
                t.name.contains("embed_tokens")
                    || t.name.contains("tok_embeddings")
                    || t.name.contains("token_embd")
            })
            .map(|t| t.name.as_str())
            .ok_or_else(|| RealizarError::FormatError {
                reason: "APR: embedding tensor not found".to_string(),
            })?;

        let embed_tensor =
            apr.find_tensor(embed_name)
                .ok_or_else(|| RealizarError::FormatError {
                    reason: "APR: embedding tensor not found".to_string(),
                })?;
        let embed_start = data_offset + embed_tensor.offset as usize;
        let embed_end = embed_start + embed_tensor.size as usize;
        if embed_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "APR: embedding tensor extends past EOF".to_string(),
            });
        }
        let embed_data = &data[embed_start..embed_end];
        let embed_dtype = Some(embed_tensor.dtype.as_str());
        let token_embedding: Vec<f32> = match embed_dtype {
            Some("F32") => embed_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Some("Q4_K") => {
                // Dequantize Q4_K embeddings
                crate::quantize::dequantize_q4_k(embed_data)?
            },
            Some(dtype) => {
                return Err(RealizarError::FormatError {
                    reason: format!("APR: unsupported embedding dtype: {dtype}"),
                });
            },
            None => {
                return Err(RealizarError::FormatError {
                    reason: "APR: embedding tensor dtype not found".to_string(),
                });
            },
        };

        // Build layers
        // Supports both HuggingFace names (SafeTensors→APR path) and GGUF names
        let mut layers = Vec::with_capacity(num_layers);
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        for layer_idx in 0..num_layers {
            // HF names (primary, from SafeTensors→APR pipeline)
            let hf_q = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
            let hf_k = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
            let hf_v = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");
            let hf_o = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
            let hf_gate = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
            let hf_up = format!("model.layers.{layer_idx}.mlp.up_proj.weight");
            let hf_down = format!("model.layers.{layer_idx}.mlp.down_proj.weight");
            let hf_attn_norm = format!("model.layers.{layer_idx}.input_layernorm.weight");
            let hf_ffn_norm = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");

            // GGUF names (fallback, from GGUF→APR path)
            let gguf_q = format!("blk.{layer_idx}.attn_q.weight");
            let gguf_k = format!("blk.{layer_idx}.attn_k.weight");
            let gguf_v = format!("blk.{layer_idx}.attn_v.weight");
            let gguf_o = format!("blk.{layer_idx}.attn_output.weight");
            let gguf_gate = format!("blk.{layer_idx}.ffn_gate.weight");
            let gguf_up = format!("blk.{layer_idx}.ffn_up.weight");
            let gguf_down = format!("blk.{layer_idx}.ffn_down.weight");
            let gguf_attn_norm = format!("blk.{layer_idx}.attn_norm.weight");
            let gguf_ffn_norm = format!("blk.{layer_idx}.ffn_norm.weight");

            // Q/K/V weights (try HF first, then GGUF)
            let q_weight = make_tensor(&[&hf_q, &gguf_q], hidden_dim, hidden_dim)?;
            let k_weight = make_tensor(&[&hf_k, &gguf_k], hidden_dim, kv_dim)?;
            let v_weight = make_tensor(&[&hf_v, &gguf_v], hidden_dim, kv_dim)?;

            let qkv_weight = OwnedQKVWeights::Separate {
                q: q_weight,
                k: k_weight,
                v: v_weight,
            };

            // QKV biases (Qwen2 has separate Q, K, V biases — concatenate for CUDA)
            let hf_q_bias = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
            let hf_k_bias = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
            let hf_v_bias = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");
            let qkv_bias = try_get_f32(&hf_q_bias).and_then(|q_b| {
                let k_b = try_get_f32(&hf_k_bias)?;
                let v_b = try_get_f32(&hf_v_bias)?;
                let mut combined = Vec::with_capacity(q_b.len() + k_b.len() + v_b.len());
                combined.extend_from_slice(&q_b);
                combined.extend_from_slice(&k_b);
                combined.extend_from_slice(&v_b);
                Some(combined)
            });

            // O projection
            let o_weight = make_tensor(&[&hf_o, &gguf_o], hidden_dim, hidden_dim)?;

            // FFN weights
            let ffn_gate_weight =
                make_tensor(&[&hf_gate, &gguf_gate], hidden_dim, intermediate_dim)?;
            let ffn_up_weight = make_tensor(&[&hf_up, &gguf_up], hidden_dim, intermediate_dim)?;
            let ffn_down_weight =
                make_tensor(&[&hf_down, &gguf_down], intermediate_dim, hidden_dim)?;

            // Norm weights (F32)
            let attn_norm_weight = get_f32_tensor(&[&hf_attn_norm, &gguf_attn_norm])?;
            let ffn_norm_weight = get_f32_tensor(&[&hf_ffn_norm, &gguf_ffn_norm])?;

            layers.push(OwnedQuantizedLayer {
                attn_norm_weight,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: o_weight,
                attn_output_bias: None,
                ffn_norm_weight: Some(ffn_norm_weight),
                ffn_norm_bias: None,
                ffn_gate_weight: Some(ffn_gate_weight),
                ffn_gate_bias: None,
                ffn_up_weight,
                ffn_up_bias: None,
                ffn_down_weight,
                ffn_down_bias: None,
            });
        }

        // Output norm
        let output_norm_weight = get_f32_tensor(&["model.norm.weight", "output_norm.weight"])?;

        // LM head (try HF name first, then GGUF)
        let lm_head_weight =
            make_tensor(&["lm_head.weight", "output.weight"], hidden_dim, vocab_size)?;

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }
}
