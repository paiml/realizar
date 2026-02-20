
/// Tensor lookup helper for APR v2 format parsing.
///
/// Bundles raw file data and parsed tensor index for clean access patterns.
/// Replaces closure-based lookups with method calls.
struct AprTensorLookup<'a> {
    data: &'a [u8],
    tensors: &'a std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
}

/// Dequantize Q4K/Q5K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q4k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && dims[1] % 256 != 0 {
        dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
            dequant_q4k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q4_k_apr(tensor_data, num_elements)
    }
}

/// Dequantize Q6K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q6k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && dims[1] % 256 != 0 {
        dequant_perrow(tensor_data, dims, 256, 210, |block, out| {
            dequant_q6k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q6_k_apr(tensor_data, num_elements)
    }
}

/// Dequantize a single tensor's raw bytes based on GGML dtype.
/// GH-191: Match on GGML dtype values written by converter.
fn dequant_by_dtype(tensor_data: &[u8], dims: &[usize], dtype: u8) -> Vec<f32> {
    match dtype {
        12 | 13 => dequant_q4k_tensor(tensor_data, dims),
        14 => dequant_q6k_tensor(tensor_data, dims),
        // GH-239: dtype=8 is ambiguous — either GGML Q8_0 or APR Q4 native.
        8 => {
            let num_elements: usize = dims.iter().product();
            let num_blocks = num_elements.div_ceil(32);
            if tensor_data.len() == num_blocks * 34 {
                dequantize_q8_0_apr(tensor_data, num_elements)
            } else {
                dequantize_apr_q4_native(tensor_data, num_elements)
            }
        },
        9 => {
            let num_elements: usize = dims.iter().product();
            dequantize_apr_q8_native(tensor_data, num_elements)
        },
        1 => tensor_data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => tensor_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

impl AprTensorLookup<'_> {
    /// Retrieve the raw byte slice for a named tensor, or None if missing/out-of-bounds.
    fn raw_bytes(&self, name: &str) -> Option<(&[u8], &Vec<usize>, u8)> {
        self.tensors.get(name).and_then(|(offset, size, dims, dtype)| {
            let end = offset + size;
            if end > self.data.len() {
                return None;
            }
            Some((&self.data[*offset..end], dims, *dtype))
        })
    }

    /// Extract tensor as f32 values (with dequantization for Q4K/Q5K/Q6K/Q8_0/APR Q4/Q8).
    fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.raw_bytes(name)
            .map(|(tensor_data, dims, dtype)| dequant_by_dtype(tensor_data, dims, dtype))
    }

    /// Extract raw Q4K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype values (12=Q4_K, 13=Q5_K).
    fn get_q4k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 12 && dtype != 13 { return None; }
            Some(data.to_vec())
        })
    }

    /// Extract raw Q6K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype value 14=Q6_K.
    fn get_q6k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 14 { return None; }
            Some(data.to_vec())
        })
    }
}

/// HF and GGUF naming prefixes for a single layer.
struct LayerPrefixes {
    hf: String,
    gguf: String,
}

impl LayerPrefixes {
    fn new(layer_idx: usize) -> Self {
        Self {
            hf: format!("model.layers.{layer_idx}"),
            gguf: format!("blk.{layer_idx}"),
        }
    }
}

impl AprTensorLookup<'_> {
    /// Look up a tensor by HF name first, then GGUF name, returning f32 values.
    fn get_hf_or_gguf(&self, hf_name: &str, gguf_name: &str) -> Option<Vec<f32>> {
        self.get_f32(hf_name).or_else(|| self.get_f32(gguf_name))
    }

    /// Load fused or separate QKV weight for a layer.
    fn load_qkv_weight(
        &self,
        pfx: &LayerPrefixes,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        if let Some(qkv) = self.get_f32(&format!("{}.self_attn.qkv_proj.weight", pfx.hf)) {
            return qkv;
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.weight", pfx.hf),
            &format!("{}.attn_q.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.weight", pfx.hf),
            &format!("{}.attn_k.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.weight", pfx.hf),
            &format!("{}.attn_v.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(&q);
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        qkv
    }

    /// Load fused or separate QKV bias for a layer (optional, for Qwen models).
    fn load_qkv_bias(&self, pfx: &LayerPrefixes) -> Option<Vec<f32>> {
        if let Some(fused) = self.get_f32(&format!("{}.self_attn.qkv_proj.bias", pfx.hf)) {
            return Some(fused);
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.bias", pfx.hf),
            &format!("{}.attn_q.bias", pfx.gguf),
        );
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.bias", pfx.hf),
            &format!("{}.attn_k.bias", pfx.gguf),
        );
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.bias", pfx.hf),
            &format!("{}.attn_v.bias", pfx.gguf),
        );
        match (&q, &k, &v) {
            (Some(q), Some(k), Some(v)) => {
                let mut bias = Vec::with_capacity(q.len() + k.len() + v.len());
                bias.extend_from_slice(q);
                bias.extend_from_slice(k);
                bias.extend_from_slice(v);
                Some(bias)
            },
            _ => None,
        }
    }

    /// Load Q4K/Q6K raw bytes for all layer weights (PMAT-103).
    fn load_quantized_layer_weights(&self, pfx: &LayerPrefixes) -> Q4KLayerWeights {
        let get_q4k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q4k(hf).or_else(|| self.get_q4k(gguf))
        };
        let get_q6k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q6k(hf).or_else(|| self.get_q6k(gguf))
        };
        Q4KLayerWeights {
            qkv_weight: None,
            attn_q_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.q_proj.weight", pfx.hf),
                &format!("{}.attn_q.weight", pfx.gguf),
            ),
            attn_k_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.k_proj.weight", pfx.hf),
                &format!("{}.attn_k.weight", pfx.gguf),
            ),
            attn_v_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_v_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_output_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ),
            ffn_gate_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            ),
            ffn_up_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
            ffn_down_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_down_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_up_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
        }
    }
}

/// Check if a `Q4KLayerWeights` has any quantized data.
fn has_quantized_data(w: &Q4KLayerWeights) -> bool {
    w.attn_q_weight.is_some()
        || w.attn_k_weight.is_some()
        || w.attn_output_weight.is_some()
        || w.ffn_gate_weight.is_some()
        || w.ffn_up_weight.is_some()
        || w.ffn_down_weight.is_some()
        || w.ffn_down_weight_q6k.is_some()
        || w.ffn_up_weight_q6k.is_some()
        || w.attn_v_weight_q6k.is_some()
}

impl AprTransformer {
    /// Build transformer layers from APR tensor data.
    ///
    /// Loads Q/K/V weights, attention output, FFN weights, and norms for each layer.
    /// Also extracts Q4K/Q6K raw bytes for fused kernel inference.
    fn build_apr_layers(
        lookup: &AprTensorLookup<'_>,
        num_layers: usize,
        hidden_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        debug_enabled: bool,
    ) -> (Vec<AprTransformerLayer>, Option<Vec<Q4KLayerWeights>>) {
        let mut layers = Vec::with_capacity(num_layers);
        let mut q4k_layer_weights: Vec<Q4KLayerWeights> = Vec::with_capacity(num_layers);
        let mut has_any_q4k = false;

        for i in 0..num_layers {
            let pfx = LayerPrefixes::new(i);

            let qkv_weight = lookup.load_qkv_weight(&pfx, hidden_dim, kv_dim);
            let qkv_bias = lookup.load_qkv_bias(&pfx);
            let attn_output = lookup.get_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
            let attn_norm = lookup.get_hf_or_gguf(
                &format!("{}.input_layernorm.weight", pfx.hf),
                &format!("{}.attn_norm.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![1.0; hidden_dim]);
            let ffn_norm = lookup.get_hf_or_gguf(
                &format!("{}.post_attention_layernorm.weight", pfx.hf),
                &format!("{}.ffn_norm.weight", pfx.gguf),
            );
            let ffn_gate = lookup.get_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            );
            let ffn_up = lookup.get_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = lookup.get_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            let quant_weights = lookup.load_quantized_layer_weights(&pfx);
            if has_quantized_data(&quant_weights) {
                has_any_q4k = true;
            }
            q4k_layer_weights.push(quant_weights);

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: attn_output,
                attn_output_bias: None,
                ffn_gate_weight: ffn_gate,
                ffn_gate_bias: None,
                ffn_up_weight: ffn_up,
                ffn_up_bias: None,
                ffn_down_weight: ffn_down,
                ffn_down_bias: None,
                ffn_norm_weight: ffn_norm,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
            });
        }

        let q4k_layers = if has_any_q4k {
            if debug_enabled {
                eprintln!("[DEBUG] Loaded Q4K raw bytes for fused kernel inference");
            }
            Some(q4k_layer_weights)
        } else {
            None
        };

        (layers, q4k_layers)
    }
}
