impl AprTransformer {

    /// Load APR transformer from bytes
    ///
    /// Parses APR v2 format from memory buffer.
    pub fn from_apr_bytes(data: &[u8]) -> Result<Self> {
        // Check minimum size for header
        if data.len() < 64 {
            return Err(RealizarError::FormatError {
                reason: format!("APR file too small: {} bytes (need 64)", data.len()),
            });
        }

        // Check magic - first 3 bytes must be "APR", 4th byte is version (0, '1', or '2')
        let magic = data.get(0..4).expect("APR file validated to have at least 64 bytes above");
        if magic[0..3] != *b"APR" || (magic[3] != 0 && magic[3] != b'1' && magic[3] != b'2') {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: {:?}, expected APR followed by version byte",
                    String::from_utf8_lossy(magic)
                ),
            });
        }

        // Parse header
        // APR header layout:
        //   0-3: Magic "APR\0"
        //   4-5: Version major.minor
        //   6-7: Flags
        //   8-11: Tensor count
        //   12-19: Metadata offset
        //   20-23: Metadata size
        //   24-31: Tensor index offset
        //   32-39: Data offset
        //   40-43: Checksum
        //   44-63: Reserved

        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]) as usize;
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]) as usize;
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]) as usize;

        // Parse metadata (JSON)
        let metadata_end = metadata_offset + metadata_size;
        if metadata_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "Metadata extends beyond file".to_string(),
            });
        }

        let metadata_json = &data[metadata_offset..metadata_end];
        let metadata: serde_json::Value = serde_json::from_slice(metadata_json).unwrap_or_default();

        // Extract architecture info from metadata
        let hidden_dim = metadata
            .get("hidden_size")
            .or_else(|| metadata.get("hidden_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(64) as usize;

        let num_layers = metadata
            .get("num_hidden_layers")
            .or_else(|| metadata.get("num_layers"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;

        let num_heads = metadata
            .get("num_attention_heads")
            .or_else(|| metadata.get("num_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(4) as usize;

        let num_kv_heads = metadata
            .get("num_key_value_heads")
            .or_else(|| metadata.get("num_kv_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(num_heads as u64) as usize;

        let vocab_size = metadata
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32000) as usize;

        let intermediate_dim = metadata
            .get("intermediate_size")
            .or_else(|| metadata.get("intermediate_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        let rope_theta = metadata
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10000.0) as f32;

        let rms_norm_eps = metadata
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;

        // PMAT-238 FIX: APR files use "context_length" (not "max_position_embeddings").
        // Check all aliases matching aprender's AprMetadata serde aliases.
        let max_position = metadata
            .get("max_position_embeddings")
            .or_else(|| metadata.get("context_length"))
            .or_else(|| metadata.get("max_seq_len"))
            .or_else(|| metadata.get("n_ctx"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2048) as usize;

        // PMAT-125: Extract architecture from metadata (was missing, defaulted to "unknown")
        // Check "architecture" first (APR v2 standard), then "model_type" (fallback)
        let architecture = metadata
            .get("architecture")
            .or_else(|| metadata.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_lowercase)
            .filter(|s| s != "auto" && !s.is_empty()) // "Auto" is not a valid architecture
            .unwrap_or_else(|| "unknown".to_string());

        let config = AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: max_position,
            rope_theta,
            eps: rms_norm_eps,
        };

        if std::env::var("REALIZE_DEBUG").is_ok() {
            eprintln!("[DEBUG] AprTransformerConfig: hidden_dim={}, num_layers={}, num_heads={}, num_kv_heads={}, vocab_size={}, intermediate_dim={}",
                hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_dim);
        }

        // Parse tensor index
        // APR v2 TensorIndexEntry format:
        //   - name_len (2 bytes) + name (variable)
        //   - dtype (1 byte)
        //   - ndim (1 byte) + dims (8 bytes each)
        //   - offset (8 bytes)
        //   - size (8 bytes)
        // Tuple: (offset, size, dims, dtype)
        let mut tensors: std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)> =
            std::collections::BTreeMap::new();

        let mut pos = tensor_index_offset;
        for _ in 0..tensor_count {
            if pos + 4 > data.len() {
                break;
            }

            // Read tensor name length and name
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            if pos + name_len + 18 > data.len() {
                break;
            }

            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            // Read dtype (1 byte)
            let dtype = data[pos];
            pos += 1;

            // Read ndim (1 byte)
            let ndim = data[pos] as usize;
            pos += 1;

            // Read dimensions (8 bytes each)
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                if pos + 8 > data.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                dims.push(dim);
                pos += 8;
            }

            // Read offset (8 bytes)
            if pos + 16 > data.len() {
                break;
            }
            let offset = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            // Read size (8 bytes)
            let size = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            tensors.insert(name, (data_offset + offset, size, dims, dtype));
        }

        // Tensor lookup helper (replaces closures for decomposition)
        let lookup = AprTensorLookup { data, tensors: &tensors };

        // Debug: print available tensor names (only when REALIZE_DEBUG is set)
        let debug_enabled = std::env::var("REALIZE_DEBUG").is_ok();
        if debug_enabled {
            eprintln!("[DEBUG] APR v2 tensor count: {tensor_count}");
            eprintln!("[DEBUG] Available tensor names (first 10):");
            for (i, (name, (offset, size, dims, dtype))) in tensors.iter().enumerate() {
                if i < 10 {
                    eprintln!(
                        "  {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}"
                    );
                }
            }
        }

        // GH-187: Enhanced logging for embedding tensor (ALWAYS log, not just debug)
        // Transposition mismatch is the most common root cause for incorrect output
        let embed_names = [
            "model.embed_tokens.weight",
            "token_embd.weight",
            "tok_embeddings.weight",
        ];
        let mut embed_dims: Option<Vec<usize>> = None;
        for name in &embed_names {
            if let Some((_offset, _size, dims, _dtype)) = tensors.get(*name) {
                embed_dims = Some(dims.clone());
                // GH-253: Only log in debug mode (was causing selftest parity failures)
                if debug_enabled {
                    eprintln!(
                        "[APR-LOAD] Embedding tensor '{}': dims={:?}, expected [vocab={}, hidden={}]",
                        name, dims, vocab_size, hidden_dim
                    );
                }
                break;
            }
        }

        // Try to load token embedding - FAIL FAST if not found (no silent zeros)
        let token_embedding_raw = lookup.get_f32("model.embed_tokens.weight")
            .or_else(|| lookup.get_f32("token_embd.weight"))
            .or_else(|| lookup.get_f32("tok_embeddings.weight"))
            .ok_or_else(|| RealizarError::FormatError {
                reason: "FATAL: No embedding tensor found. Tried: model.embed_tokens.weight, \
                        token_embd.weight, tok_embeddings.weight. APR file may be corrupt or \
                        use unsupported tensor naming convention."
                    .to_string(),
            })?;

        // GH-208 FIX: Do NOT transpose embedding data
        // GGML data layout: data[i0 + i1*ne0] for shape [ne0, ne1]
        // For embedding [ne0=hidden, ne1=vocab]: data[h + v*hidden] = row-major [vocab, hidden]
        // Token v's embedding = data[v*hidden .. (v+1)*hidden] - ALREADY CORRECT
        // The GH-187 transpose was WRONG - it corrupted embeddings (correlation 0.001 instead of 1.0)
        // See: contracts/tensor-layout-v1.yaml, compare_embed example
        let token_embedding = token_embedding_raw;
        if debug_enabled {
            if let Some(ref dims) = embed_dims {
                eprintln!(
                    "[APR-LOAD] Embedding dims={:?}, using raw data (no transpose needed)",
                    dims
                );
            }
        }

        // GH-187: Sanity check - verify embedding produces non-garbage for token 0
        if token_embedding.len() >= hidden_dim {
            let first_embed = &token_embedding[0..hidden_dim];
            let all_zero = first_embed.iter().all(|&x| x == 0.0);
            let has_nan = first_embed.iter().any(|x| x.is_nan());
            let has_inf = first_embed.iter().any(|x| x.is_infinite());
            if all_zero {
                eprintln!(
                    "[APR-LOAD] WARNING: Token 0 embedding is all zeros - possible load failure"
                );
            }
            if has_nan || has_inf {
                eprintln!(
                    "[APR-LOAD] ERROR: Token 0 embedding contains NaN/Inf - data corruption!"
                );
            }
            if debug_enabled {
                eprintln!(
                    "[APR-LOAD] Token 0 embedding sample: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    first_embed[0],
                    first_embed.get(1).unwrap_or(&0.0),
                    first_embed.get(2).unwrap_or(&0.0),
                    first_embed.get(3).unwrap_or(&0.0),
                    first_embed.get(4).unwrap_or(&0.0)
                );
            }
        }

        // GH-253: Only log embedding size in debug mode
        if debug_enabled {
            eprintln!(
                "[APR-LOAD] Embedding loaded: {} elements (vocab={} x hidden={})",
                token_embedding.len(),
                vocab_size,
                hidden_dim
            );
        }

        // Load output norm
        let output_norm_weight = lookup.get_f32("model.norm.weight")
            .or_else(|| lookup.get_f32("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden_dim]);

        // GH-187/GH-253: Log lm_head tensor info only in debug mode
        if debug_enabled {
            for name in &["lm_head.weight", "output.weight"] {
                if let Some((_offset, _size, dims, dtype)) = tensors.get(*name) {
                    eprintln!(
                        "[APR-LOAD] LM head tensor '{}': dims={:?}, dtype={}, expected [vocab={}, hidden={}]",
                        name, dims, dtype, vocab_size, hidden_dim
                    );
                    break;
                }
            }
        }

        // Load LM head - FAIL FAST if not found (no silent zeros)
        // For tied embeddings (common in Qwen, LLaMA models), use embed_tokens as fallback
        let lm_head_raw =
            lookup.get_f32("lm_head.weight").or_else(|| lookup.get_f32("output.weight"));
        let (lm_head_raw, used_tied_weights) = if let Some(lm_head) = lm_head_raw {
            (lm_head, false)
        } else {
            // Weight tying: use embedding weights for lm_head
            if debug_enabled {
                eprintln!("[APR-LOAD] No lm_head found, trying tied embedding weights");
            }
            let tied = lookup.get_f32("model.embed_tokens.weight")
                .or_else(|| lookup.get_f32("token_embd.weight"));
            if let Some(t) = tied {
                (t, true)
            } else {
                return Err(RealizarError::FormatError {
                    reason: "FATAL: No lm_head tensor found and no embedding for weight tying. \
                            Tried: lm_head.weight, output.weight, model.embed_tokens.weight, \
                            token_embd.weight. APR file may be corrupt."
                        .to_string(),
                });
            }
        };
        if used_tied_weights && debug_enabled {
            eprintln!("[APR-LOAD] Using tied weights: embedding -> lm_head");
        }

        // GH-187: lm_head is used for matmul (not lookup), so GGML layout is correct
        // lm_head: y = x @ W where W is [hidden_dim, vocab_size] in GGML convention
        // This matches fused_q4k_parallel_matvec(weights, x, in_dim=hidden, out_dim=vocab)
        // NO transposition needed for lm_head (unlike embedding)
        let lm_head_weight = lm_head_raw;
        if debug_enabled {
            eprintln!(
                "[APR-LOAD] LM head loaded: {} elements (hidden={} x vocab={})",
                lm_head_weight.len(),
                hidden_dim,
                vocab_size
            );
        }

        // PMAT-103: Load lm_head Q4K/Q6K raw bytes for fused kernel inference
        let lm_head_weight_q4k =
            lookup.get_q4k("lm_head.weight").or_else(|| lookup.get_q4k("output.weight"));
        let lm_head_weight_q6k =
            lookup.get_q6k("lm_head.weight").or_else(|| lookup.get_q6k("output.weight"));
        // GH-187/GH-253: Log quantization path only in debug mode
        if debug_enabled {
            if let Some(ref bytes) = lm_head_weight_q4k {
                eprintln!(
                    "[APR-LOAD] LM head using Q4K fused kernel ({} bytes)",
                    bytes.len()
                );
            } else if let Some(ref bytes) = lm_head_weight_q6k {
                eprintln!(
                    "[APR-LOAD] LM head using Q6K fused kernel ({} bytes)",
                    bytes.len()
                );
            } else {
                eprintln!("[APR-LOAD] LM head using F32 matmul (no Q4K/Q6K found)");
            }
        }

        // Compute KV dimension from config
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Load per-layer weights (extracted for decomposition)
        let (layers, q4k_layers) = Self::build_apr_layers(
            &lookup, num_layers, hidden_dim, kv_dim, intermediate_dim, debug_enabled,
        );

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers,
            lm_head_weight_q6k,
            lm_head_weight_q4k,
        })
    }
}

include!("mod_part_02_part_03_load.rs");
