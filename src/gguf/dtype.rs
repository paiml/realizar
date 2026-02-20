
/// Convert GGML qtype to APR dtype string
fn apr_qtype_to_dtype(qtype: u32) -> &'static str {
    match qtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        30 => "BF16",
        _ => "F32",
    }
}

/// Convert APR dtype string to byte for binary tensor entry
/// GH-191 FIX: Use GGML dtype values directly so they match TensorEntry::from_binary reader.
fn apr_dtype_to_byte(dtype: &str) -> u8 {
    match dtype {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 30,    // GGML BF16 type
        "Q4_0" => 2,     // GGML type 2
        "Q4_1" => 3,     // GGML type 3
        "Q5_0" => 6,     // GGML type 6
        "Q5_1" => 7,     // GGML type 7
        "Q8_0" => 8,     // GGML type 8
        "Q8_1" => 9,     // GGML type 9
        "Q2_K" => 10,    // GGML type 10
        "Q3_K" => 11,    // GGML type 11
        "Q4_K" => 12,    // GGML type 12
        "Q5_K" => 13,    // GGML type 13
        "Q6_K" => 14,    // GGML type 14
        "IQ2_XXS" => 16, // GGML type 16
        "IQ2_XS" => 17,  // GGML type 17
        _ => {
            eprintln!(
                "WARN: Unknown dtype '{}' in dtype_to_byte, writing as F32",
                dtype
            );
            0
        },
    }
}

/// Write a single tensor entry to APR binary index format
fn write_apr_tensor_entry(
    name: &str,
    dtype: &str,
    shape: &[usize],
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let mut entry = Vec::new();

    // Name: 2-byte length + bytes
    let name_bytes = name.as_bytes();
    entry.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    entry.extend_from_slice(name_bytes);

    // Dtype: 1 byte
    entry.push(apr_dtype_to_byte(dtype));

    // Shape: 1-byte ndim + 8-byte dims
    entry.push(shape.len() as u8);
    for &dim in shape {
        entry.extend_from_slice(&(dim as u64).to_le_bytes());
    }

    // Offset and size: 8 bytes each
    entry.extend_from_slice(&offset.to_le_bytes());
    entry.extend_from_slice(&size.to_le_bytes());

    entry
}

impl OwnedQuantizedModel {

    /// Serialize model to APR format with quantized weights preserved
    ///
    /// Creates a valid .apr file that can be loaded via `from_apr()`.
    /// Quantization types (Q4_K, Q6_K, etc.) are preserved in the tensor dtypes.
    ///
    /// # Returns
    ///
    /// Raw bytes in APR v2 format
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    // serde_json::json!() uses infallible unwrap
    #[allow(clippy::disallowed_methods)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_apr_bytes(&self) -> Result<Vec<u8>> {
        use crate::apr::{ALIGNMENT, HEADER_SIZE, MAGIC};

        // Collect all tensors
        let tensors = self.collect_apr_model_tensors();

        // Build metadata JSON
        let metadata = serde_json::json!({
            "model_type": "transformer_lm",
            "architecture": self.config.architecture,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "intermediate_size": self.config.intermediate_dim,
            "rms_norm_eps": self.config.eps,
            "rope_theta": self.config.rope_theta,
            "context_length": self.config.context_length,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize metadata: {e}"),
            })?;
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Build tensor index and data
        let mut tensor_index_bytes: Vec<u8> = Vec::new();
        let mut tensor_data_bytes: Vec<u8> = Vec::new();

        for (name, dtype, shape, data) in &tensors {
            // Align tensor data to 64 bytes
            let padding = (ALIGNMENT - (tensor_data_bytes.len() % ALIGNMENT)) % ALIGNMENT;
            tensor_data_bytes.extend(std::iter::repeat_n(0u8, padding));

            let offset = tensor_data_bytes.len() as u64;
            let size = data.len() as u64;

            tensor_index_bytes.extend(write_apr_tensor_entry(
                name, dtype, shape, offset, size,
            ));

            tensor_data_bytes.extend_from_slice(data);
        }

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

        // Build header
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags (quantized = bit 0)
        header[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset.to_le_bytes());
        // checksum at 40-43 (leave as 0 for now)

        // Combine all parts
        let total_size =
            HEADER_SIZE + metadata_padded_len + tensor_index_bytes.len() + tensor_data_bytes.len();
        let mut result = Vec::with_capacity(total_size);
        result.extend_from_slice(&header);
        result.extend_from_slice(&metadata_bytes);
        result.resize(HEADER_SIZE + metadata_padded_len, 0); // pad metadata
        result.extend_from_slice(&tensor_index_bytes);
        result.extend_from_slice(&tensor_data_bytes);

        Ok(result)
    }

    /// Collect all model tensors as (name, dtype, shape, data) tuples for APR serialization
    #[allow(clippy::cast_possible_truncation)]
    fn collect_apr_model_tensors(&self) -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
        let mut tensors = Vec::new();

        // Token embedding (F32)
        let embed_bytes: Vec<u8> = self
            .token_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.push((
            "token_embd.weight".to_string(),
            "F32".to_string(),
            vec![self.config.vocab_size, self.config.hidden_dim],
            embed_bytes,
        ));

        // Layers
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let kv_dim = self.config.num_kv_heads * head_dim;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.collect_apr_layer_tensors(&mut tensors, layer_idx, layer, kv_dim);
        }

        // Output norm (F32)
        let output_norm_bytes: Vec<u8> = self
            .output_norm_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.push((
            "output_norm.weight".to_string(),
            "F32".to_string(),
            vec![self.config.hidden_dim],
            output_norm_bytes,
        ));

        // LM head (quantized)
        tensors.push((
            "output.weight".to_string(),
            apr_qtype_to_dtype(self.lm_head_weight.qtype).to_string(),
            vec![self.config.vocab_size, self.config.hidden_dim],
            self.lm_head_weight.data.clone(),
        ));

        tensors
    }

    /// Collect tensors for a single transformer layer
    fn collect_apr_layer_tensors(
        &self,
        tensors: &mut Vec<(String, String, Vec<usize>, Vec<u8>)>,
        layer_idx: usize,
        layer: &OwnedQuantizedLayer,
        kv_dim: usize,
    ) {
        // Attention norm (F32)
        let norm_bytes: Vec<u8> = layer
            .attn_norm_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.push((
            format!("blk.{layer_idx}.attn_norm.weight"),
            "F32".to_string(),
            vec![self.config.hidden_dim],
            norm_bytes,
        ));

        // QKV weights (quantized)
        match &layer.qkv_weight {
            OwnedQKVWeights::Separate { q, k, v } => {
                tensors.push((
                    format!("blk.{layer_idx}.attn_q.weight"),
                    apr_qtype_to_dtype(q.qtype).to_string(),
                    vec![self.config.hidden_dim, self.config.hidden_dim],
                    q.data.clone(),
                ));
                tensors.push((
                    format!("blk.{layer_idx}.attn_k.weight"),
                    apr_qtype_to_dtype(k.qtype).to_string(),
                    vec![kv_dim, self.config.hidden_dim],
                    k.data.clone(),
                ));
                tensors.push((
                    format!("blk.{layer_idx}.attn_v.weight"),
                    apr_qtype_to_dtype(v.qtype).to_string(),
                    vec![kv_dim, self.config.hidden_dim],
                    v.data.clone(),
                ));
            },
            OwnedQKVWeights::Fused(t) => {
                tensors.push((
                    format!("blk.{layer_idx}.attn_qkv.weight"),
                    apr_qtype_to_dtype(t.qtype).to_string(),
                    vec![t.out_dim, t.in_dim],
                    t.data.clone(),
                ));
            },
        }

        // Output projection (quantized)
        tensors.push((
            format!("blk.{layer_idx}.attn_output.weight"),
            apr_qtype_to_dtype(layer.attn_output_weight.qtype).to_string(),
            vec![self.config.hidden_dim, self.config.hidden_dim],
            layer.attn_output_weight.data.clone(),
        ));

        // FFN norm (F32)
        if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            let norm_bytes: Vec<u8> = ffn_norm.iter().flat_map(|f| f.to_le_bytes()).collect();
            tensors.push((
                format!("blk.{layer_idx}.ffn_norm.weight"),
                "F32".to_string(),
                vec![self.config.hidden_dim],
                norm_bytes,
            ));
        }

        // FFN weights (quantized)
        if let Some(ref gate) = layer.ffn_gate_weight {
            tensors.push((
                format!("blk.{layer_idx}.ffn_gate.weight"),
                apr_qtype_to_dtype(gate.qtype).to_string(),
                vec![self.config.intermediate_dim, self.config.hidden_dim],
                gate.data.clone(),
            ));
        }

        tensors.push((
            format!("blk.{layer_idx}.ffn_up.weight"),
            apr_qtype_to_dtype(layer.ffn_up_weight.qtype).to_string(),
            vec![self.config.intermediate_dim, self.config.hidden_dim],
            layer.ffn_up_weight.data.clone(),
        ));

        tensors.push((
            format!("blk.{layer_idx}.ffn_down.weight"),
            apr_qtype_to_dtype(layer.ffn_down_weight.qtype).to_string(),
            vec![self.config.hidden_dim, self.config.intermediate_dim],
            layer.ffn_down_weight.data.clone(),
        ));
    }
}

include!("embedding.rs");
include!("loader_apr_quantized.rs");
