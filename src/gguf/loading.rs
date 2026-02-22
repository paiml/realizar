/// Load a quantized tensor from APR format, trying multiple names.
///
/// Handles APR native q8/q4 formats by dequantizing to f32.
/// For Conv1D architectures, transposes weights to [out, in] layout.
fn apr_load_quantized_tensor(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
    in_dim: usize,
    out_dim: usize,
    transpose: bool,
) -> Result<OwnedQuantizedTensor> {
    use crate::apr::MappedAprModel;

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
    let raw = &data[start..end];
    let dtype = tensor.dtype.as_str();
    let num_elements = in_dim * out_dim;

    match dtype {
        "q8" => {
            let mut f32_data = crate::apr::dequant::dequantize_apr_q8(raw, num_elements);
            if transpose {
                f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
            }
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(OwnedQuantizedTensor {
                data: f32_bytes,
                in_dim,
                out_dim,
                qtype: 0,
            })
        },
        "q4" => {
            let mut f32_data = crate::apr::dequant::dequantize_apr_q4(raw, num_elements);
            if transpose {
                f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
            }
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(OwnedQuantizedTensor {
                data: f32_bytes,
                in_dim,
                out_dim,
                qtype: 0,
            })
        },
        _ => {
            let qtype = MappedAprModel::dtype_to_qtype(dtype);
            Ok(OwnedQuantizedTensor {
                data: raw.to_vec(),
                in_dim,
                out_dim,
                qtype,
            })
        },
    }
}

/// Load an F32 tensor from APR format, trying multiple names.
fn apr_load_f32_tensor(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
) -> Result<Vec<f32>> {
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
}

/// Try loading an optional F32 bias tensor from APR format.
fn apr_try_load_f32(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    name: &str,
) -> Option<Vec<f32>> {
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
}

/// Infer vocab_size from APR metadata or embedding tensor shape.
fn apr_infer_vocab_size(apr: &crate::apr::MappedAprModel) -> usize {
    match apr.metadata.vocab_size {
        Some(v) if v > 0 => v,
        _ => apr
            .tensors
            .iter()
            .find(|t| {
                t.name.contains("embed_tokens")
                    || t.name.contains("tok_embeddings")
                    || t.name.contains("token_embd")
            })
            .and_then(|t| t.shape.first().copied())
            .unwrap_or(151936),
    }
}

impl OwnedQuantizedModel {
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
        let vocab_size = apr_infer_vocab_size(apr);

        let architecture = apr
            .metadata
            .architecture
            .clone()
            .unwrap_or_else(|| "qwen2".to_string());
        let constraints = crate::gguf::ArchConstraints::from_architecture(&architecture);
        let config = GGUFConfig {
            architecture,
            constraints,
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
            explicit_head_dim: None,
            bos_token_id: apr.metadata.get_embedded_bos_token_id(),
        };

        // GH-279: Contract gate — validate architecture and dimensions before loading weights
        let _proof = crate::contract_gate::validate_model_load_basic(
            &config.architecture,
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_dim,
            config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        // GH-278: Detect Conv1D layout from contract (not string matching)
        let transpose = config.constraints.needs_transpose();

        // Load token embeddings
        let token_embedding =
            Self::load_apr_token_embedding(apr, data, data_offset, vocab_size, hidden_dim)?;

        // Build layers
        let head_dim = config.head_dim();
        let kv_dim = config.kv_dim();
        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            layers.push(Self::load_apr_layer(
                apr,
                data,
                data_offset,
                layer_idx,
                hidden_dim,
                kv_dim,
                intermediate_dim,
                transpose,
            )?);
        }

        // Output norm
        let output_norm_weight =
            apr_load_f32_tensor(apr, data, data_offset, &["model.norm.weight", "output_norm.weight"])?;
        let output_norm_bias = apr_try_load_f32(apr, data, data_offset, "model.norm.bias");

        // LM head (try HF name first, then GGUF)
        let lm_head_weight = apr_load_quantized_tensor(
            apr, data, data_offset,
            &["lm_head.weight", "output.weight"],
            hidden_dim, vocab_size, transpose,
        )?;
        let lm_head_bias = apr_try_load_f32(apr, data, data_offset, "lm_head.bias");

        // GH-278: Load learned position embeddings (GPT-2 style)
        let position_embedding =
            apr_try_load_f32(apr, data, data_offset, "model.position_embedding.weight");

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
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
        })
    }

    /// Load token embeddings from APR format.
    fn load_apr_token_embedding(
        apr: &crate::apr::MappedAprModel,
        data: &[u8],
        data_offset: usize,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
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

        let embed_tensor = apr.find_tensor(embed_name).ok_or_else(|| RealizarError::FormatError {
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
        dequantize_embedding(embed_data, embed_tensor.dtype.as_str(), vocab_size * hidden_dim)
    }

    /// Load a single transformer layer from APR format.
    #[allow(clippy::too_many_arguments)]
    fn load_apr_layer(
        apr: &crate::apr::MappedAprModel,
        data: &[u8],
        data_offset: usize,
        layer_idx: usize,
        hidden_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        transpose: bool,
    ) -> Result<OwnedQuantizedLayer> {
        // HF names (primary, from SafeTensors->APR pipeline)
        let hf_q = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
        let hf_k = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
        let hf_v = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");
        let hf_o = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let hf_gate = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
        let hf_up = format!("model.layers.{layer_idx}.mlp.up_proj.weight");
        let hf_down = format!("model.layers.{layer_idx}.mlp.down_proj.weight");
        let hf_attn_norm = format!("model.layers.{layer_idx}.input_layernorm.weight");
        let hf_ffn_norm = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");

        // GGUF names (fallback, from GGUF->APR path)
        let gguf_q = format!("blk.{layer_idx}.attn_q.weight");
        let gguf_k = format!("blk.{layer_idx}.attn_k.weight");
        let gguf_v = format!("blk.{layer_idx}.attn_v.weight");
        let gguf_o = format!("blk.{layer_idx}.attn_output.weight");
        let gguf_gate = format!("blk.{layer_idx}.ffn_gate.weight");
        let gguf_up = format!("blk.{layer_idx}.ffn_up.weight");
        let gguf_down = format!("blk.{layer_idx}.ffn_down.weight");
        let gguf_attn_norm = format!("blk.{layer_idx}.attn_norm.weight");
        let gguf_ffn_norm = format!("blk.{layer_idx}.ffn_norm.weight");

        let q_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_q, &gguf_q], hidden_dim, hidden_dim, transpose)?;
        let k_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_k, &gguf_k], hidden_dim, kv_dim, transpose)?;
        let v_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_v, &gguf_v], hidden_dim, kv_dim, transpose)?;

        let qkv_weight = OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        };

        // QKV biases (Qwen2 has separate Q, K, V biases — concatenate for CUDA)
        let hf_q_bias = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
        let hf_k_bias = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
        let hf_v_bias = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");
        let qkv_bias = apr_try_load_f32(apr, data, data_offset, &hf_q_bias).and_then(|q_b| {
            let k_b = apr_try_load_f32(apr, data, data_offset, &hf_k_bias)?;
            let v_b = apr_try_load_f32(apr, data, data_offset, &hf_v_bias)?;
            let mut combined = Vec::with_capacity(q_b.len() + k_b.len() + v_b.len());
            combined.extend_from_slice(&q_b);
            combined.extend_from_slice(&k_b);
            combined.extend_from_slice(&v_b);
            Some(combined)
        });

        let o_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_o, &gguf_o], hidden_dim, hidden_dim, transpose)?;

        // FFN weights (gate is optional — GPT-2 has no SwiGLU gate)
        let ffn_gate_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_gate, &gguf_gate], hidden_dim, intermediate_dim, transpose).ok();
        let ffn_up_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_up, &gguf_up], hidden_dim, intermediate_dim, transpose)?;
        let ffn_down_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_down, &gguf_down], intermediate_dim, hidden_dim, transpose)?;

        // Norm weights (F32)
        let attn_norm_weight = apr_load_f32_tensor(apr, data, data_offset, &[&hf_attn_norm, &gguf_attn_norm])?;
        let ffn_norm_weight = apr_load_f32_tensor(apr, data, data_offset, &[&hf_ffn_norm, &gguf_ffn_norm]).ok();

        // GH-278: Load biases (GPT-2/phi-2 style models have biases on all projections)
        let hf_attn_norm_bias = format!("model.layers.{layer_idx}.input_layernorm.bias");
        let hf_ffn_norm_bias = format!("model.layers.{layer_idx}.post_attention_layernorm.bias");
        let hf_o_bias = format!("model.layers.{layer_idx}.self_attn.o_proj.bias");
        let hf_up_bias = format!("model.layers.{layer_idx}.mlp.up_proj.bias");
        let hf_down_bias = format!("model.layers.{layer_idx}.mlp.down_proj.bias");

        Ok(OwnedQuantizedLayer {
            attn_norm_weight,
            attn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_attn_norm_bias),
            qkv_weight,
            qkv_bias,
            attn_output_weight: o_weight,
            attn_output_bias: apr_try_load_f32(apr, data, data_offset, &hf_o_bias),
            ffn_norm_weight,
            ffn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_ffn_norm_bias),
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: apr_try_load_f32(apr, data, data_offset, &hf_up_bias),
            ffn_down_weight,
            ffn_down_bias: apr_try_load_f32(apr, data, data_offset, &hf_down_bias),
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
    }
}
