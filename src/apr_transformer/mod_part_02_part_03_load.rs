
/// Tensor lookup helper for APR v2 format parsing.
///
/// Bundles raw file data and parsed tensor index for clean access patterns.
/// Replaces closure-based lookups with method calls.
struct AprTensorLookup<'a> {
    data: &'a [u8],
    tensors: &'a std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
}

impl AprTensorLookup<'_> {
    /// Extract tensor as f32 values (with dequantization for Q4K/Q5K/Q6K/Q8_0/APR Q4/Q8).
    fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.tensors.get(name).map(|(offset, size, dims, dtype)| {
            let end = offset + size;
            if end > self.data.len() {
                return Vec::new();
            }
            let tensor_data = &self.data[*offset..end];

            // GH-191 FIX: Match on GGML dtype values written by converter.
            match dtype {
                // Q4_K (GGML type 12)
                12 => {
                    // GH-202 FIX: Handle per-row padding for 2D tensors.
                    if dims.len() == 2 && dims[1] % 256 != 0 {
                        dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
                            dequant_q4k_block(block, out);
                        })
                    } else {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q4_k_apr(tensor_data, num_elements)
                    }
                },
                // Q5_K (GGML type 13) - use Q4_K dequant (compatible layout)
                13 => {
                    if dims.len() == 2 && dims[1] % 256 != 0 {
                        dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
                            dequant_q4k_block(block, out);
                        })
                    } else {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q4_k_apr(tensor_data, num_elements)
                    }
                },
                // Q6_K (GGML type 14)
                14 => {
                    // GH-202 FIX: Handle per-row padding for 2D tensors.
                    if dims.len() == 2 && dims[1] % 256 != 0 {
                        dequant_perrow(tensor_data, dims, 256, 210, |block, out| {
                            dequant_q6k_block(block, out);
                        })
                    } else {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q6_k_apr(tensor_data, num_elements)
                    }
                },
                // GH-239: dtype=8 is ambiguous — either GGML Q8_0 or APR Q4 native.
                8 => {
                    let num_elements: usize = dims.iter().product();
                    let num_blocks = num_elements.div_ceil(32);
                    if tensor_data.len() == num_blocks * 34 {
                        // GGML Q8_0: [f16 scale (2B) + 32×i8 quants] = 34 bytes/block
                        dequantize_q8_0_apr(tensor_data, num_elements)
                    } else {
                        // APR Q4: [f16 scale (2B) + 16 nibble bytes] = 18 bytes/block
                        dequantize_apr_q4_native(tensor_data, num_elements)
                    }
                },
                // GH-239: APR Q8 native (dtype=9)
                9 => {
                    let num_elements: usize = dims.iter().product();
                    dequantize_apr_q8_native(tensor_data, num_elements)
                },
                // F16 (GGML type 1): convert f16 to f32
                1 => tensor_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                // F32 (dtype=0) or other: interpret as raw F32
                _ => tensor_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            }
        })
    }

    /// Extract raw Q4K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype values (12=Q4_K, 13=Q5_K).
    fn get_q4k(&self, name: &str) -> Option<Vec<u8>> {
        self.tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
            if *dtype != 12 && *dtype != 13 {
                return None;
            }
            let end = offset + size;
            if end > self.data.len() {
                return None;
            }
            Some(self.data[*offset..end].to_vec())
        })
    }

    /// Extract raw Q6K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype value 14=Q6_K.
    fn get_q6k(&self, name: &str) -> Option<Vec<u8>> {
        self.tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
            if *dtype != 14 {
                return None;
            }
            let end = offset + size;
            if end > self.data.len() {
                return None;
            }
            Some(self.data[*offset..end].to_vec())
        })
    }
}

impl AprTransformer {

    /// Build transformer layers from APR tensor data.
    ///
    /// Loads Q/K/V weights, attention output, FFN weights, and norms for each layer.
    /// Also extracts Q4K/Q6K raw bytes for fused kernel inference.
    #[allow(clippy::too_many_arguments)]
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
        let qkv_out_dim = hidden_dim + kv_dim + kv_dim;

        for i in 0..num_layers {
            let hf_prefix = format!("model.layers.{i}");
            let gguf_prefix = format!("blk.{i}");

            // Detect GGUF vs HF naming
            let is_gguf = lookup.tensors.contains_key(&format!("{gguf_prefix}.attn_q.weight"));

            let qkv_weight = if let Some(qkv) =
                lookup.get_f32(&format!("{hf_prefix}.self_attn.qkv_proj.weight"))
            {
                qkv
            } else {
                let q_raw = lookup.get_f32(&format!("{hf_prefix}.self_attn.q_proj.weight"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_q.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
                let k_raw = lookup.get_f32(&format!("{hf_prefix}.self_attn.k_proj.weight"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_k.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
                let v_raw = lookup.get_f32(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_v.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);

                let _ = is_gguf; // Suppress unused warning
                let mut qkv = Vec::with_capacity(qkv_out_dim * hidden_dim);
                qkv.extend_from_slice(&q_raw);
                qkv.extend_from_slice(&k_raw);
                qkv.extend_from_slice(&v_raw);
                qkv
            };

            // Get Q/K/V biases (optional, for Qwen models)
            let qkv_bias = if let Some(fused_bias) =
                lookup.get_f32(&format!("{hf_prefix}.self_attn.qkv_proj.bias"))
            {
                Some(fused_bias)
            } else {
                let q_bias = lookup.get_f32(&format!("{hf_prefix}.self_attn.q_proj.bias"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_q.bias")));
                let k_bias = lookup.get_f32(&format!("{hf_prefix}.self_attn.k_proj.bias"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_k.bias")));
                let v_bias = lookup.get_f32(&format!("{hf_prefix}.self_attn.v_proj.bias"))
                    .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_v.bias")));

                match (&q_bias, &k_bias, &v_bias) {
                    (Some(q), Some(k), Some(v)) => {
                        let mut bias = Vec::with_capacity(qkv_out_dim);
                        bias.extend_from_slice(q);
                        bias.extend_from_slice(k);
                        bias.extend_from_slice(v);
                        Some(bias)
                    },
                    _ => None,
                }
            };

            let attn_output = lookup.get_f32(&format!("{hf_prefix}.self_attn.o_proj.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_output.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);

            let attn_norm = lookup.get_f32(&format!("{hf_prefix}.input_layernorm.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.attn_norm.weight")))
                .unwrap_or_else(|| vec![1.0; hidden_dim]);

            let ffn_norm = lookup.get_f32(&format!("{hf_prefix}.post_attention_layernorm.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.ffn_norm.weight")));

            let ffn_gate = lookup.get_f32(&format!("{hf_prefix}.mlp.gate_proj.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.ffn_gate.weight")));
            let ffn_up = lookup.get_f32(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.ffn_up.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = lookup.get_f32(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| lookup.get_f32(&format!("{gguf_prefix}.ffn_down.weight")))
                .unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            // PMAT-103 FIX: Extract Q4K and Q6K raw bytes for fused kernel
            let q4k_attn_q = lookup.get_q4k(&format!("{hf_prefix}.self_attn.q_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.attn_q.weight")));
            let q4k_attn_k = lookup.get_q4k(&format!("{hf_prefix}.self_attn.k_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.attn_k.weight")));
            let q4k_attn_v = lookup.get_q4k(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.attn_v.weight")));
            let q6k_attn_v = lookup.get_q6k(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                .or_else(|| lookup.get_q6k(&format!("{gguf_prefix}.attn_v.weight")));
            let q4k_attn_output =
                lookup.get_q4k(&format!("{hf_prefix}.self_attn.o_proj.weight"))
                    .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.attn_output.weight")));
            let q4k_ffn_gate = lookup.get_q4k(&format!("{hf_prefix}.mlp.gate_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.ffn_gate.weight")));
            let q4k_ffn_up = lookup.get_q4k(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.ffn_up.weight")));
            let q4k_ffn_down = lookup.get_q4k(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| lookup.get_q4k(&format!("{gguf_prefix}.ffn_down.weight")));
            let q6k_ffn_down = lookup.get_q6k(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| lookup.get_q6k(&format!("{gguf_prefix}.ffn_down.weight")));
            let q6k_ffn_up = lookup.get_q6k(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| lookup.get_q6k(&format!("{gguf_prefix}.ffn_up.weight")));

            let has_q4k_weights = q4k_attn_q.is_some()
                || q4k_attn_k.is_some()
                || q4k_attn_output.is_some()
                || q4k_ffn_gate.is_some()
                || q4k_ffn_up.is_some()
                || q4k_ffn_down.is_some();
            let has_q6k_weights =
                q6k_ffn_down.is_some() || q6k_ffn_up.is_some() || q6k_attn_v.is_some();

            if has_q4k_weights || has_q6k_weights {
                has_any_q4k = true;
            }

            q4k_layer_weights.push(Q4KLayerWeights {
                qkv_weight: None,
                attn_q_weight: q4k_attn_q,
                attn_k_weight: q4k_attn_k,
                attn_v_weight: q4k_attn_v,
                attn_v_weight_q6k: q6k_attn_v,
                attn_output_weight: q4k_attn_output,
                ffn_gate_weight: q4k_ffn_gate,
                ffn_up_weight: q4k_ffn_up,
                ffn_down_weight: q4k_ffn_down,
                ffn_down_weight_q6k: q6k_ffn_down,
                ffn_up_weight_q6k: q6k_ffn_up,
            });

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
