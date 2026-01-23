//! Model components for transformer inference
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - KVCache: Key-Value cache for efficient autoregressive generation
//! - TransformerBlock: Single transformer layer (attention + FFN)
//! - Embedding: Token embedding layer
//! - Model: Full transformer model for inference
//! - ModelConfig: Configuration for transformer models

use crate::{
    error::{RealizarError, Result},
    generate::{sample_token, GenerationConfig},
    tensor::Tensor,
};

use super::{FeedForward, LayerNorm, Linear, MultiHeadAttention};

/// Key-Value Cache for efficient transformer inference
///
/// Stores key and value tensors from previous positions to avoid
/// recomputation during autoregressive generation. Each forward pass
/// only computes K/V for the new token and appends to the cache.
///
/// # Usage
///
/// 1. Create cache with `KVCache::new(num_layers, max_seq_len, head_dim)`
/// 2. At each generation step, call `update` with new K/V
/// 3. Use `get_key`/`get_value` to retrieve cached tensors
/// 4. Call `clear` to reset for new sequence
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Dimension per head
    head_dim: usize,
    /// Current sequence position
    current_pos: usize,
    /// Cached keys for each layer: `[num_layers][max_seq_len * head_dim]`
    keys: Vec<Vec<f32>>,
    /// Cached values for each layer: `[num_layers][max_seq_len * head_dim]`
    values: Vec<Vec<f32>>,
}

impl KVCache {
    /// Create a new KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers to cache
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `head_dim` - Dimension per attention head
    ///
    /// # Errors
    ///
    /// Returns error if any dimension is zero
    pub fn new(num_layers: usize, max_seq_len: usize, head_dim: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_layers must be > 0".to_string(),
            });
        }
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "max_seq_len must be > 0".to_string(),
            });
        }
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }

        let cache_size = max_seq_len * head_dim;
        let keys = vec![vec![0.0; cache_size]; num_layers];
        let values = vec![vec![0.0; cache_size]; num_layers];

        Ok(Self {
            num_layers,
            max_seq_len,
            head_dim,
            current_pos: 0,
            keys,
            values,
        })
    }

    /// Update cache with new key/value for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `key` - New key tensor `[head_dim]`
    /// * `value` - New value tensor `[head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds, cache is full, or tensor sizes don't match
    pub fn update(&mut self, layer: usize, key: &Tensor<f32>, value: &Tensor<f32>) -> Result<()> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }
        if self.current_pos >= self.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Cache full at position {} (max {})",
                    self.current_pos, self.max_seq_len
                ),
            });
        }

        let k_data = key.data();
        let v_data = value.data();

        if k_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Key size {} != head_dim {}", k_data.len(), self.head_dim),
            });
        }
        if v_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Value size {} != head_dim {}", v_data.len(), self.head_dim),
            });
        }

        // Copy key and value into cache at current position
        let offset = self.current_pos * self.head_dim;
        self.keys[layer][offset..offset + self.head_dim].copy_from_slice(k_data);
        self.values[layer][offset..offset + self.head_dim].copy_from_slice(v_data);

        Ok(())
    }

    /// Advance to next position after updating all layers
    ///
    /// Call this after updating all layers for the current position
    pub fn advance(&mut self) {
        if self.current_pos < self.max_seq_len {
            self.current_pos += 1;
        }
    }

    /// Get cached keys for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_key(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            // Return empty tensor with shape [0, head_dim] is invalid
            // Return [1, head_dim] with zeros for consistency
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.keys[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Get cached values for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_value(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.values[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Clear cache and reset position to 0
    pub fn clear(&mut self) {
        self.current_pos = 0;
        // Optionally zero out the cache (not strictly necessary)
        for layer in 0..self.num_layers {
            self.keys[layer].fill(0.0);
            self.values[layer].fill(0.0);
        }
    }

    /// Get current sequence position
    #[must_use]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Get number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Check if cache is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }
}

/// Transformer Block (Pre-norm architecture)
///
/// A single transformer block combining self-attention and feed-forward layers
/// with residual connections and layer normalization.
///
/// # Architecture
///
/// ```text
/// Input
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// Attention            │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// FFN                  │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
/// Output
/// ```
///
/// This is the pre-norm architecture used in `LLaMA`, GPT-NeoX, and modern transformers.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Layer normalization before attention
    attn_norm: LayerNorm,
    /// Multi-head self-attention layer with Q/K/V/O projections
    attention: MultiHeadAttention,
    /// Layer normalization before FFN
    ffn_norm: LayerNorm,
    /// Feed-forward network
    ffn: FeedForward,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
}

impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Hidden dimension (model dimension)
    /// * `num_heads` - Number of attention heads
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `eps` - Layer normalization epsilon
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `hidden_dim` is zero or not divisible by `num_heads`
    /// - `num_heads` is zero
    /// - `intermediate_dim` is zero
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }

        let attn_norm = LayerNorm::new(hidden_dim, eps)?;
        // Use standard MHA with Q/K/V/O projections
        let attention = MultiHeadAttention::mha(hidden_dim, num_heads)?;
        let ffn_norm = LayerNorm::new(hidden_dim, eps)?;
        let ffn = FeedForward::new(hidden_dim, intermediate_dim)?;

        Ok(Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            hidden_dim,
            num_heads,
        })
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape is invalid
    ///
    /// # Note
    ///
    /// This simplified implementation uses the same input for Q, K, V (self-attention).
    /// Production models would compute Q, K, V projections separately.
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Expected last dimension {}, got {}",
                    self.hidden_dim, last_dim
                ),
            });
        }

        // Pre-norm attention block
        let normed = self.attn_norm.forward(input)?;

        // Self-attention with proper Q/K/V/O projections via MultiHeadAttention
        let attn_out = self.attention.forward(&normed)?;

        // Residual connection
        let mut residual1 = Vec::with_capacity(input.data().len());
        for (i, &val) in input.data().iter().enumerate() {
            residual1.push(val + attn_out.data()[i]);
        }
        let after_attn = Tensor::from_vec(shape.to_vec(), residual1)?;

        // Pre-norm FFN block
        let normed2 = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = self.ffn.forward(&normed2)?;

        // Residual connection
        let mut residual2 = Vec::with_capacity(after_attn.data().len());
        for (i, &val) in after_attn.data().iter().enumerate() {
            residual2.push(val + ffn_out.data()[i]);
        }

        Tensor::from_vec(shape.to_vec(), residual2)
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get mutable reference to attention layer normalization
    pub fn attn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.attn_norm
    }

    /// Get mutable reference to multi-head attention
    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention {
        &mut self.attention
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get mutable reference to FFN layer normalization
    pub fn ffn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.ffn_norm
    }

    /// Get mutable reference to FFN
    pub fn ffn_mut(&mut self) -> &mut FeedForward {
        &mut self.ffn
    }
}

/// Embedding layer for converting token IDs to vectors
///
/// Maps discrete token IDs to continuous vector representations.
/// This is the first layer in a transformer model.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Embedding weights: `[vocab_size, embed_dim]`
    weights: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding layer
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary
    /// * `embed_dim` - Dimension of embedding vectors
    ///
    /// # Errors
    ///
    /// Returns error if `vocab_size` or `embed_dim` is zero
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if embed_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "embed_dim must be > 0".to_string(),
            });
        }

        let weights = vec![0.0; vocab_size * embed_dim];

        Ok(Self {
            vocab_size,
            embed_dim,
            weights,
        })
    }

    /// Look up embeddings for token IDs
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token IDs
    ///
    /// # Returns
    ///
    /// Tensor with shape `[seq_len, embed_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is out of bounds
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let mut output = Vec::with_capacity(seq_len * self.embed_dim);

        for &token_id in token_ids {
            if token_id >= self.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {token_id} out of bounds (vocab_size={})",
                        self.vocab_size
                    ),
                });
            }

            let offset = token_id * self.embed_dim;
            output.extend_from_slice(&self.weights[offset..offset + self.embed_dim]);
        }

        Tensor::from_vec(vec![seq_len, self.embed_dim], output)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get mutable access to weights for loading
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

/// Transformer Language Model
///
/// Complete transformer model for language modeling:
/// - Token embedding
/// - Stack of transformer blocks
/// - Final layer normalization
/// - Output projection (LM head)
///
/// # Architecture
///
/// ```text
/// Token IDs → Embedding → [TransformerBlock × N] → LayerNorm → Linear → Logits
/// ```
#[derive(Debug, Clone)]
pub struct Model {
    /// Token embedding layer
    embedding: Embedding,
    /// Stack of transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Final layer normalization
    final_norm: LayerNorm,
    /// Output projection (LM head)
    lm_head: Linear,
    /// Model configuration
    config: ModelConfig,
}

/// Configuration for the transformer model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Layer normalization epsilon
    pub eps: f32,
}

impl Model {
    /// Create a new transformer model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: ModelConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.eps,
            )?);
        }

        let final_norm = LayerNorm::new(config.hidden_dim, config.eps)?;
        let lm_head = Linear::new(config.hidden_dim, config.vocab_size)?;

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
            config,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits tensor with shape `[seq_len, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns error if input is invalid
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        // Embed tokens
        let mut hidden = self.embedding.forward(token_ids)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        hidden = self.final_norm.forward(&hidden)?;

        // Project to vocabulary
        self.lm_head.forward(&hidden)
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get mutable reference to embedding layer
    pub fn embedding_mut(&mut self) -> &mut Embedding {
        &mut self.embedding
    }

    /// Get mutable reference to transformer blocks
    pub fn blocks_mut(&mut self) -> &mut [TransformerBlock] {
        &mut self.blocks
    }

    /// Get mutable reference to final layer norm
    pub fn final_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.final_norm
    }

    /// Get mutable reference to LM head
    pub fn lm_head_mut(&mut self) -> &mut Linear {
        &mut self.lm_head
    }

    /// Get number of parameters in the model (approximate)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.hidden_dim;
        let block_params = self.config.num_layers
            * (
                // Attention (Q, K, V, O projections would be here in full impl)
                // For now just count layer norms and FFN
                2 * self.config.hidden_dim  // Layer norm weights
                + self.config.hidden_dim * self.config.intermediate_dim  // fc1
                + self.config.intermediate_dim * self.config.hidden_dim
                // fc2
            );
        let head_params = self.config.hidden_dim * self.config.vocab_size;

        embed_params + block_params + head_params
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Vector of generated token IDs (including prompt)
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let generated = model.generate(&[1, 2, 3], &GenerationConfig::greedy())?;
    /// ```
    pub fn generate(&self, prompt: &[usize], config: &GenerationConfig) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();
        let mut rng_state = config.seed.unwrap_or(42);

        for _ in 0..config.max_tokens {
            // Forward pass
            let logits = self.forward(&tokens)?;

            // Get logits for last position
            let seq_len = tokens.len();
            let vocab_size = self.config.vocab_size;
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data()[last_logits_start..last_logits_start + vocab_size];

            let last_logits_tensor = Tensor::from_vec(vec![vocab_size], last_logits.to_vec())?;

            // Simple LCG for random number generation
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let rng_value = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

            // Sample next token
            let next_token = sample_token(&last_logits_tensor, config, rng_value)?;

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
