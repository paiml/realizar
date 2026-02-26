
impl SafetensorsToAprConverter {
    /// Convert SafeTensors model to APR Transformer
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model.safetensors file
    ///
    /// # Returns
    ///
    /// `AprTransformer` with F32 weights ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if SafeTensors file, config.json, or required tensors are missing
    pub fn convert(model_path: &Path) -> Result<ValidatedAprTransformer> {
        // Load SafeTensors model using mmap for zero-copy access (T-QA-020)
        // This is critical for fast model loading - mmap is O(1) regardless of file size
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        // Load config.json (required for architecture info)
        let config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "safetensors_convert".to_string(),
                reason: "config.json not found (required for SafeTensors inference)".to_string(),
            }
        })?;

        Self::convert_from_source(&st_model, &config)
    }

    /// Convert a sharded SafeTensors model to APR Transformer (GH-213)
    ///
    /// # Arguments
    ///
    /// * `sharded` - Loaded sharded model (from index.json)
    /// * `config` - Model config.json
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing from any shard
    #[cfg(not(target_arch = "wasm32"))]
    pub fn convert_sharded(
        sharded: &ShardedSafeTensorsModel,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        Self::convert_from_source(sharded, config)
    }

    /// Core conversion logic shared between single-file and sharded paths
    fn convert_from_source<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        // Extract architecture parameters
        let hidden_dim = config
            .hidden_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing hidden_size".to_string(),
            })?;
        let num_layers = config
            .num_hidden_layers
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = config
            .num_attention_heads
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config
            .vocab_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing vocab_size".to_string(),
            })?;
        let intermediate_dim = config.intermediate_size.unwrap_or(hidden_dim * 4);
        let context_length = config.max_position_embeddings.unwrap_or(0);
        let architecture = config.architecture();
        // R-02 (Meyer DbC): rope_theta from config, or architecture-specific default.
        let rope_theta = config.rope_theta.unwrap_or_else(||
            crate::gguf::default_rope_theta_for_architecture(&architecture));
        let eps = config.rms_norm_eps.unwrap_or(1e-6);

        // Phase 2: Validate model dimensions at construction boundary.
        // Non-fatal: logs a warning for invalid configs but allows conversion to
        // proceed. The final ValidatedAprTransformer::validate() at the end is
        // the hard gate. This catches obvious dimension errors early.
        if let Err(e) = crate::gguf::ValidatedModelConfig::from_safetensors_config(config) {
            eprintln!(
                "[Phase2-WARN] SafeTensors config validation: {e} — proceeding with conversion"
            );
        }

        // GH-278: Log Qwen3.5 detection with hybrid attention info
        if config.is_hybrid_attention() {
            let layer_count = config.layer_types.as_ref().map_or(0, Vec::len);
            let linear_count = config.layer_types.as_ref().map_or(0, |t| {
                t.iter()
                    .filter(|l| *l == "linear" || *l == "linear_attention")
                    .count()
            });
            eprintln!(
                "[GH-278] Hybrid attention model detected: {}/{} linear layers, head_dim={:?}",
                linear_count,
                layer_count,
                config.head_dim,
            );
        }

        // Build transformer config
        let apr_config = AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            eos_token_id: config.eos_token_id,
            // GH-278: Hybrid attention fields
            explicit_head_dim: config.head_dim,
            layer_types: config.layer_types.clone(),
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        };

        // Extract embeddings (HuggingFace: model.embed_tokens.weight, GGUF: token_embd.weight)
        let token_embedding = Self::get_tensor_with_fallback_generic(
            source,
            "model.embed_tokens.weight",
            "token_embd.weight",
        )?;

        // Extract output norm (HuggingFace: model.norm.weight, GGUF: output_norm.weight)
        let output_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            "model.norm.weight",
            "output_norm.weight",
        )?;

        // F-GT-002 FIX: Check tie_word_embeddings config FIRST, not just tensor existence
        // When tie_word_embeddings=true, HuggingFace may store a placeholder lm_head.weight
        // that's all zeros - we MUST use the embedding matrix instead!
        let use_tied_embeddings = config.tie_word_embeddings.unwrap_or(false);

        let lm_head_weight = if use_tied_embeddings {
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        } else if Self::has_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")
        {
            let raw =
                Self::get_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")?;
            Self::transpose_weight(&raw, vocab_size, hidden_dim)
        } else {
            // Fallback: assume tied if no lm_head tensor exists
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        };

        // Extract layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // GH-278: Dispatch to linear layer extractor for Gated Delta Net layers
            let is_linear = config
                .layer_types
                .as_ref()
                .and_then(|lt| lt.get(i))
                .is_some_and(|t| t == "linear" || t == "linear_attention");

            let layer = if is_linear {
                Self::extract_linear_layer_generic(
                    source,
                    i,
                    hidden_dim,
                    intermediate_dim,
                    config,
                )?
            } else {
                Self::extract_layer_generic(
                    source,
                    i,
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    intermediate_dim,
                )?
            };
            layers.push(layer);
        }

        let transformer = AprTransformer {
            config: apr_config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        // PMAT-235: Validate ALL tensors at construction time
        ValidatedAprTransformer::validate(transformer).map_err(Into::into)
    }

    /// Extract a single transformer layer from SafeTensors (MappedSafeTensorsModel)
    fn extract_layer(
        st_model: &MappedSafeTensorsModel,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        Self::extract_layer_generic(
            st_model,
            layer_idx,
            hidden_dim,
            num_heads,
            num_kv_heads,
            intermediate_dim,
        )
    }

    /// Extract a single transformer layer from any TensorSource (GH-213)
    fn extract_layer_generic<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        // Support both HuggingFace and GGUF naming conventions
        let hf_prefix = format!("model.layers.{layer_idx}");
        let gguf_prefix = format!("blk.{layer_idx}");

        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("{gguf_prefix}.attn_norm.weight"),
        )?;

        let q_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.q_proj.weight"),
            &format!("{gguf_prefix}.attn_q.weight"),
        )?;
        let k_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.k_proj.weight"),
            &format!("{gguf_prefix}.attn_k.weight"),
        )?;
        let v_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.v_proj.weight"),
            &format!("{gguf_prefix}.attn_v.weight"),
        )?;

        let head_dim = hidden_dim / num_heads;
        let kv_dim = head_dim * num_kv_heads;
        let qkv_weight =
            Self::concat_qkv_transposed(&q_weight, &k_weight, &v_weight, hidden_dim, kv_dim);

        // QKV bias (optional)
        let qkv_bias = Self::try_concat_qkv_bias_dual_generic(
            source,
            &hf_prefix,
            &gguf_prefix,
            hidden_dim,
            kv_dim,
        );

        let attn_output_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.o_proj.weight"),
            &format!("{gguf_prefix}.attn_output.weight"),
        )?;
        let attn_output_weight = Self::transpose_weight(&attn_output_raw, hidden_dim, hidden_dim);

        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("{gguf_prefix}.ffn_norm.weight"),
        )?;

        let ffn_gate_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.gate_proj.weight"),
            &format!("{gguf_prefix}.ffn_gate.weight"),
        )?;
        let ffn_gate_weight = Self::transpose_weight(&ffn_gate_raw, intermediate_dim, hidden_dim);

        let ffn_up_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.up_proj.weight"),
            &format!("{gguf_prefix}.ffn_up.weight"),
        )?;
        let ffn_up_weight = Self::transpose_weight(&ffn_up_raw, intermediate_dim, hidden_dim);

        let ffn_down_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.down_proj.weight"),
            &format!("{gguf_prefix}.ffn_down.weight"),
        )?;
        let ffn_down_weight = Self::transpose_weight(&ffn_down_raw, hidden_dim, intermediate_dim);

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            // GH-279: QK norm weights (Qwen3 per-head RMSNorm)
            attn_q_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_q_norm.weight")))
                .ok(),
            attn_k_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_k_norm.weight")))
                .ok(),
            // GH-278: Linear attention weights extracted separately in extract_linear_layer_generic
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
        })
    }

    /// GH-278: Extract a single Gated Delta Net (linear attention) transformer layer
    ///
    /// Qwen3.5 linear layers have combined projections different from standard attention:
    /// - `in_proj_qkvz`: Combined Q+K+V+Z → split into qkv_weight + z_weight
    /// - `in_proj_ba`: Combined Beta+Alpha → split into b_weight + a_weight
    /// - `out_proj`: Output projection (not `o_proj`)
    /// - `conv1d`, `A_log`, `dt_bias`, `norm`: GDN-specific parameters
    fn extract_linear_layer_generic<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        config: &SafetensorsConfig,
    ) -> Result<AprTransformerLayer> {
        let hf_prefix = format!("model.layers.{layer_idx}");

        // --- Attention layer norm (same as standard layers) ---
        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("blk.{layer_idx}.attn_norm.weight"),
        )?;

        // --- GDN combined projections ---
        // in_proj_qkvz: [q_dim + k_dim + v_dim + z_dim, hidden_dim]
        let in_proj_qkvz = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.in_proj_qkvz.weight"))?;

        // Compute split dimensions from config
        let key_head_dim = config.linear_key_head_dim.unwrap_or(128);
        let value_head_dim = config.linear_value_head_dim.unwrap_or(128);
        let num_key_heads = config.linear_num_key_heads.unwrap_or(16);
        let num_value_heads = config.linear_num_value_heads.unwrap_or(32);
        let key_dim = num_key_heads * key_head_dim;
        let value_dim = num_value_heads * value_head_dim;
        // in_proj_qkvz layout: [Q(key_dim) | K(key_dim) | V(value_dim) | Z(value_dim), hidden_dim]
        let qkvz_out_dim = 2 * key_dim + 2 * value_dim;

        // Validate combined projection size
        let expected_qkvz = qkvz_out_dim * hidden_dim;
        if in_proj_qkvz.len() != expected_qkvz {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "GH-278: in_proj_qkvz size mismatch at layer {layer_idx}: \
                     expected {expected_qkvz} (qkvz_out={qkvz_out_dim} * hidden={hidden_dim}), \
                     got {}",
                    in_proj_qkvz.len()
                ),
            });
        }

        // Split QKVZ along the output dimension (row-major: each row is hidden_dim)
        let q_end = key_dim * hidden_dim;
        let k_end = q_end + key_dim * hidden_dim;
        let v_end = k_end + value_dim * hidden_dim;
        // z occupies the rest: [v_end .. qkvz_out_dim * hidden_dim]
        let q_weight = &in_proj_qkvz[..q_end];
        let k_weight = &in_proj_qkvz[q_end..k_end];
        let v_weight = &in_proj_qkvz[k_end..v_end];
        let z_weight = in_proj_qkvz[v_end..].to_vec();

        // Concatenate Q+K+V into qkv_weight (same format as standard layers)
        let qkv_weight = Self::concat_qkv(q_weight, k_weight, v_weight);

        // in_proj_ba: [num_v_heads + num_v_heads, hidden_dim] = [2*num_v_heads, hidden_dim]
        let in_proj_ba = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.in_proj_ba.weight"))?;

        let ba_split = num_value_heads * hidden_dim;
        let b_weight = in_proj_ba[..ba_split].to_vec();
        let a_weight = in_proj_ba[ba_split..].to_vec();

        // out_proj: [hidden_dim, value_dim] — GDN uses out_proj, not o_proj
        let out_proj_raw = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.out_proj.weight"))?;
        let attn_output_weight = Self::transpose_weight(&out_proj_raw, hidden_dim, value_dim);

        // Conv1D weight: HF stores as [conv_dim, 1, kernel_size], squeeze middle dim
        let conv1d_raw = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.conv1d.weight"))?;
        // Already flattened to [conv_dim * kernel_size] by get_tensor_auto
        let conv1d_weight = conv1d_raw;

        // A_log: [num_v_heads] — parameter, no .weight suffix
        let a_log = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.A_log"))?;

        // dt_bias: [num_v_heads] — parameter, no .weight suffix
        let dt_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.dt_bias"))?;

        // Gated RMSNorm weight
        let norm_weight = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.norm.weight"))?;

        // --- FFN weights (same as standard layers: SwiGLU MLP) ---
        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("blk.{layer_idx}.ffn_norm.weight"),
        )?;

        let ffn_gate_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.gate_proj.weight"),
            &format!("blk.{layer_idx}.ffn_gate.weight"),
        )?;
        let ffn_gate_weight = Self::transpose_weight(&ffn_gate_raw, intermediate_dim, hidden_dim);

        let ffn_up_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.up_proj.weight"),
            &format!("blk.{layer_idx}.ffn_up.weight"),
        )?;
        let ffn_up_weight = Self::transpose_weight(&ffn_up_raw, intermediate_dim, hidden_dim);

        let ffn_down_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.down_proj.weight"),
            &format!("blk.{layer_idx}.ffn_down.weight"),
        )?;
        let ffn_down_weight = Self::transpose_weight(&ffn_down_raw, hidden_dim, intermediate_dim);

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias: None,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            // GH-278: Gated Delta Net weights
            linear_attn_z_weight: Some(z_weight),
            linear_attn_b_weight: Some(b_weight),
            linear_attn_a_weight: Some(a_weight),
            linear_attn_conv1d_weight: Some(conv1d_weight),
            linear_attn_a_log: Some(a_log),
            linear_attn_dt_bias: Some(dt_bias),
            linear_attn_norm_weight: Some(norm_weight),
        })
    }

    /// Pass through weight in matvec-optimal [out_dim, in_dim] format
    ///
    /// PMAT-095 FIX: HuggingFace stores Linear weights as [out_features, in_features]
    /// which is EXACTLY what trueno's matvec needs! Previous implementation transposed
    /// twice (here and in matmul), causing O(n²) overhead per forward pass.
    ///
    /// Now we keep HuggingFace format directly - no transposition needed.
    #[allow(clippy::unused_self)]
    pub fn transpose_weight(weight: &[f32], _out_dim: usize, _in_dim: usize) -> Vec<f32> {
        // PMAT-095: Keep [out_dim, in_dim] format - no transposition!
        // This eliminates the 75x performance gap vs GGUF.
        weight.to_vec()
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (matvec-optimal)
    ///
    /// PMAT-095 FIX: Keep [out_dim, in_dim] format from HuggingFace.
    /// For QKV, we concatenate along the output dimension:
    /// - Q: [hidden_dim, hidden_dim]
    /// - K: [kv_dim, hidden_dim]
    /// - V: [kv_dim, hidden_dim]
    ///
    /// Result: [hidden_dim + kv_dim + kv_dim, hidden_dim] in row-major
    pub fn concat_qkv_transposed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        _hidden_dim: usize,
        _kv_dim: usize,
    ) -> Vec<f32> {
        // PMAT-095: Simple concatenation - weights are already in optimal layout
        // Concatenate [Q; K; V] along output dimension
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (legacy, no transpose)
    fn concat_qkv(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Try to concatenate Q, K, V biases if they exist
    fn try_concat_qkv_bias(
        st_model: &MappedSafeTensorsModel,
        prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
            .ok()?;
        let k_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
            .ok()?;
        let v_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
            .ok()?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Try to concatenate Q, K, V biases with dual naming support
    fn try_concat_qkv_bias_dual(
        st_model: &MappedSafeTensorsModel,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        Self::try_concat_qkv_bias_dual_generic(st_model, hf_prefix, gguf_prefix, hidden_dim, kv_dim)
    }

    /// Generic version of QKV bias concatenation (GH-213)
    fn try_concat_qkv_bias_dual_generic<S: TensorSource>(
        source: &S,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_q.bias"))
                    .ok()
            })?;
        let k_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_k.bias"))
                    .ok()
            })?;
        let v_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.v_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_v.bias"))
                    .ok()
            })?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Get tensor with fallback to alternative naming conventions
    ///
    /// Tries HuggingFace naming first, then GGUF-style naming, then bare name.
    /// This enables loading SafeTensors files regardless of their origin.
    ///
    /// GH-196: Also tries stripping `model.` prefix for APR canonical names,
    /// and adds diagnostic tensor name listing on failure.
    fn get_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        Self::get_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor lookup with fallback naming (GH-213)
    fn get_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        // Try HuggingFace name first (e.g., "model.norm.weight")
        if let Ok(t) = source.get_tensor_auto(hf_name) {
            return Ok(t);
        }
        // Try GGUF name (e.g., "output_norm.weight")
        if let Ok(t) = source.get_tensor_auto(gguf_name) {
            return Ok(t);
        }
        // Try bare name without "model." prefix (APR canonical names)
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        if bare_name != hf_name {
            if let Ok(t) = source.get_tensor_auto(bare_name) {
                return Ok(t);
            }
        }

        // Diagnostic: list available tensor names for debugging
        let available = source.tensor_names();
        let sample: Vec<&str> = available.iter().take(5).copied().collect();
        Err(RealizarError::UnsupportedOperation {
            operation: "get_tensor_auto".to_string(),
            reason: format!(
                "Tensor not found with names: '{}', '{}', or '{}'. \
                 Available tensors ({} total): {:?}{}",
                hf_name,
                gguf_name,
                bare_name,
                available.len(),
                sample,
                if available.len() > 5 { ", ..." } else { "" }
            ),
        })
    }

    /// Check if tensor exists with either naming convention
    fn has_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        Self::has_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor existence check (GH-213)
    fn has_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        if source.has_tensor(hf_name) || source.has_tensor(gguf_name) {
            return true;
        }
        // GH-196: Also check bare name without "model." prefix
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        bare_name != hf_name && source.has_tensor(bare_name)
    }

    /// Get optional tensor with fallback naming
    fn get_optional_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Option<Vec<f32>> {
        st_model
            .get_tensor_auto(hf_name)
            .ok()
            .or_else(|| st_model.get_tensor_auto(gguf_name).ok())
            .or_else(|| {
                // GH-196: Try bare name without "model." prefix
                let bare_name = hf_name.strip_prefix("model.")?;
                st_model.get_tensor_auto(bare_name).ok()
            })
    }
}
