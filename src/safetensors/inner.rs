
impl ValidatedAprTransformer {
    /// Validate all tensors in an `AprTransformer`
    ///
    /// This is the ONLY way to create a `ValidatedAprTransformer`.
    /// Every tensor is validated using the existing newtype gates:
    /// - `ValidatedEmbedding` for token_embedding
    /// - `ValidatedWeight` for weight matrices
    /// - `ValidatedVector` for norm weights and biases
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` identifying the first tensor that fails.
    pub fn validate(
        transformer: AprTransformer,
    ) -> std::result::Result<Self, ContractValidationError> {
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;
        let head_dim = if config.num_heads > 0 {
            hidden_dim / config.num_heads
        } else {
            hidden_dim
        };
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        // === Global tensors ===

        // token_embedding: [vocab_size * hidden_dim]
        ValidatedEmbedding::new(transformer.token_embedding.clone(), vocab_size, hidden_dim)
            .map_err(|mut e| {
                e.tensor_name = "token_embedding".to_string();
                e
            })?;

        // output_norm_weight: [hidden_dim]
        ValidatedVector::new(
            transformer.output_norm_weight.clone(),
            hidden_dim,
            "output_norm_weight",
        )?;

        // output_norm_bias (optional)
        if let Some(ref bias) = transformer.output_norm_bias {
            ValidatedVector::new(bias.clone(), hidden_dim, "output_norm_bias")?;
        }

        // lm_head_weight: [vocab_size * hidden_dim]
        ValidatedWeight::new(
            transformer.lm_head_weight.clone(),
            vocab_size,
            hidden_dim,
            "lm_head_weight",
        )?;

        // lm_head_bias (optional)
        if let Some(ref bias) = transformer.lm_head_bias {
            ValidatedVector::new(bias.clone(), vocab_size, "lm_head_bias")?;
        }

        // === Per-layer tensors ===
        for (i, layer) in transformer.layers.iter().enumerate() {
            // attn_norm_weight: [hidden_dim]
            ValidatedVector::new(
                layer.attn_norm_weight.clone(),
                hidden_dim,
                &format!("layers.{i}.attn_norm_weight"),
            )?;

            // attn_norm_bias (optional)
            if let Some(ref bias) = layer.attn_norm_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.attn_norm_bias"),
                )?;
            }

            // qkv_weight: [qkv_out_dim * hidden_dim]
            ValidatedWeight::new(
                layer.qkv_weight.clone(),
                qkv_out_dim,
                hidden_dim,
                &format!("layers.{i}.qkv_weight"),
            )?;

            // qkv_bias (optional)
            if let Some(ref bias) = layer.qkv_bias {
                ValidatedVector::new(bias.clone(), qkv_out_dim, &format!("layers.{i}.qkv_bias"))?;
            }

            // attn_output_weight: [hidden_dim * hidden_dim]
            ValidatedWeight::new(
                layer.attn_output_weight.clone(),
                hidden_dim,
                hidden_dim,
                &format!("layers.{i}.attn_output_weight"),
            )?;

            // attn_output_bias (optional)
            if let Some(ref bias) = layer.attn_output_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.attn_output_bias"),
                )?;
            }

            // ffn_gate_weight (optional): [intermediate_dim * hidden_dim]
            if let Some(ref w) = layer.ffn_gate_weight {
                ValidatedWeight::new(
                    w.clone(),
                    intermediate_dim,
                    hidden_dim,
                    &format!("layers.{i}.ffn_gate_weight"),
                )?;
            }

            // ffn_gate_bias (optional)
            if let Some(ref bias) = layer.ffn_gate_bias {
                ValidatedVector::new(
                    bias.clone(),
                    intermediate_dim,
                    &format!("layers.{i}.ffn_gate_bias"),
                )?;
            }

            // ffn_up_weight: [intermediate_dim * hidden_dim]
            ValidatedWeight::new(
                layer.ffn_up_weight.clone(),
                intermediate_dim,
                hidden_dim,
                &format!("layers.{i}.ffn_up_weight"),
            )?;

            // ffn_up_bias (optional)
            if let Some(ref bias) = layer.ffn_up_bias {
                ValidatedVector::new(
                    bias.clone(),
                    intermediate_dim,
                    &format!("layers.{i}.ffn_up_bias"),
                )?;
            }

            // ffn_down_weight: [hidden_dim * intermediate_dim]
            ValidatedWeight::new(
                layer.ffn_down_weight.clone(),
                hidden_dim,
                intermediate_dim,
                &format!("layers.{i}.ffn_down_weight"),
            )?;

            // ffn_down_bias (optional)
            if let Some(ref bias) = layer.ffn_down_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_down_bias"),
                )?;
            }

            // ffn_norm_weight (optional): [hidden_dim]
            if let Some(ref w) = layer.ffn_norm_weight {
                ValidatedVector::new(
                    w.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_norm_weight"),
                )?;
            }

            // ffn_norm_bias (optional)
            if let Some(ref bias) = layer.ffn_norm_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_norm_bias"),
                )?;
            }
        }

        Ok(Self { inner: transformer })
    }

    /// Access the inner transformer
    #[must_use]
    pub fn transformer(&self) -> &AprTransformer {
        &self.inner
    }

    /// Consume and return the inner transformer
    #[must_use]
    pub fn into_inner(self) -> AprTransformer {
        self.inner
    }

    /// Access the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.inner.config
    }
}

impl std::ops::Deref for ValidatedAprTransformer {
    type Target = AprTransformer;
    fn deref(&self) -> &AprTransformer {
        &self.inner
    }
}
