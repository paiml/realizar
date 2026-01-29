//! APR Transformer Configuration Types (PMAT-802)
//!
//! Configuration structs for APR transformer:
//! - AprKVCache: KV cache for efficient autoregressive generation
//! - GenerateConfig: Generation parameters
//! - AprTransformerConfig: Model architecture configuration
//! - AprTransformerLayer: Per-layer weights
//! - Q4KLayerWeights: Q4K quantized layer weights

use serde::{Deserialize, Serialize};

// ============================================================================

/// KV Cache for efficient autoregressive generation (Y4)
///
/// Pre-allocates storage for keys and values to avoid allocations during decode.
/// Each layer has separate K and V caches stored contiguously.
///
/// # Memory Layout
///
/// For each layer: `[K_pos0, K_pos1, ..., K_posN, V_pos0, V_pos1, ..., V_posN]`
/// where each K/V entry has shape `[num_kv_heads * head_dim]`.
#[derive(Debug, Clone)]
pub struct AprKVCache {
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum context length (pre-allocated capacity)
    capacity: usize,
    /// Current sequence length (positions filled)
    len: usize,
    /// True if a position is currently being appended (layers 0..N-1 have written)
    in_progress: bool,
    /// K cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    k_cache: Vec<Vec<f32>>,
    /// V cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    v_cache: Vec<Vec<f32>>,
}

impl AprKVCache {
    /// Create a new KV cache with pre-allocated capacity
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration
    ///
    /// # Returns
    ///
    /// Empty KV cache with capacity for full context length
    #[must_use]
    pub fn new(config: &AprTransformerConfig) -> Self {
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.hidden_dim / config.num_heads;
        let capacity = config.context_length;

        // Pre-allocate full capacity for each layer
        let kv_size = capacity * num_kv_heads * head_dim;
        let k_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();
        let v_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            capacity,
            len: 0,
            in_progress: false,
            k_cache,
            v_cache,
        }
    }

    /// Get current sequence length (number of cached positions)
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get pre-allocated capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get number of KV heads
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append K and V for a single position
    ///
    /// When called with `layer == num_layers - 1` (last layer), this automatically
    /// increments `self.len` so that `get()` returns the newly appended data.
    /// Tests that only use layer 0 should call `advance()` after append.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `k` - Key tensor `[num_kv_heads * head_dim]`
    /// * `v` - Value tensor `[num_kv_heads * head_dim]`
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or cache is full
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert!(self.len < self.capacity, "KV cache is full");

        let kv_size = self.num_kv_heads * self.head_dim;
        let offset = self.len * kv_size;

        // Copy K and V into pre-allocated storage
        self.k_cache[layer][offset..offset + kv_size].copy_from_slice(k);
        self.v_cache[layer][offset..offset + kv_size].copy_from_slice(v);

        // Mark that we have in-progress data (so get() includes it)
        self.in_progress = true;

        // F-REGR-231 FIX: Increment len only on LAST layer to ensure:
        // 1. All layers write to the same offset (correct for single token)
        // 2. get() immediately sees new data after last layer appends
        // 3. No manual advance() calls needed in production code
        // Note: Tests using only layer 0 should call advance() manually.
        if layer == self.num_layers - 1 {
            self.len += 1;
            self.in_progress = false;
        }
    }

    /// Advance the cache position manually.
    ///
    /// Usually not needed - `append()` auto-advances after the last layer.
    /// Only use this if you need to advance without appending all layers (e.g., in tests).
    pub fn advance(&mut self) {
        self.len += 1;
        self.in_progress = false;
    }

    /// Get cached K and V for a layer
    ///
    /// If `in_progress` is true, returns data up to `len + 1` positions to include
    /// data appended by earlier layers in the current forward pass.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (K cache slice, V cache slice) containing all cached positions
    #[must_use]
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        let kv_size = self.num_kv_heads * self.head_dim;
        // Include in-progress position if any layer has appended
        let effective_len = self.len + (self.in_progress as usize);
        let used_size = effective_len * kv_size;

        (
            &self.k_cache[layer][..used_size],
            &self.v_cache[layer][..used_size],
        )
    }

    /// Clear the cache (reset to empty without deallocating)
    pub fn clear(&mut self) {
        self.len = 0;
        self.in_progress = false;
        // No need to zero memory - will be overwritten on next append
    }
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (optional)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Enable trace output (default: false)
    pub trace: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            trace: false,
        }
    }
}

/// Configuration for APR Transformer models
///
/// Mirrors `GGUFConfig` for compatibility but is serializable to APR format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AprTransformerConfig {
    /// Model architecture name (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding/hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta for position encoding
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl Default for AprTransformerConfig {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }
}

/// Weights for a single transformer layer (all F32)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformerLayer {
    /// Attention norm weight [hidden_dim]
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional) [hidden_dim]
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [hidden_dim, 3*hidden_dim]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (optional) [3*hidden_dim]
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight [hidden_dim, hidden_dim]
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias (optional) [hidden_dim]
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate weight for SwiGLU (optional) [hidden_dim, intermediate_dim]
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate bias (optional) [intermediate_dim]
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight [hidden_dim, intermediate_dim]
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias (optional) [intermediate_dim]
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight [intermediate_dim, hidden_dim]
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias (optional) [hidden_dim]
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (optional) [hidden_dim]
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional) [hidden_dim]
    pub ffn_norm_bias: Option<Vec<f32>>,
}

impl AprTransformerLayer {
    /// Create an empty layer with given dimensions (non-GQA: num_kv_heads == num_heads)
    pub fn empty(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }
    }

    /// Create an empty layer with GQA dimensions (num_kv_heads < num_heads)
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden dimension (num_heads * head_dim)
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (< num_heads for GQA)
    /// * `intermediate_dim` - FFN intermediate dimension
    pub fn empty_gqa(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        // QKV weight: [hidden_dim, Q_dim + K_dim + V_dim] = [hidden_dim, hidden_dim + 2*kv_dim]
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * qkv_out_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }
    }

    /// Get total number of parameters in this layer
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.attn_norm_weight.len();
        count += self.attn_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.qkv_weight.len();
        count += self.qkv_bias.as_ref().map_or(0, Vec::len);
        count += self.attn_output_weight.len();
        count += self.attn_output_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_up_weight.len();
        count += self.ffn_up_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_down_weight.len();
        count += self.ffn_down_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_bias.as_ref().map_or(0, Vec::len);
        count
    }
}

/// Q4K/Q6K raw weights for fused kernel inference (F-GPU-130)
///
/// When present, matmul operations use fused kernels (matmul_q4k_f32, matmul_q6k_f32)
/// instead of the F32 path, avoiding full dequantization overhead.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Q4KLayerWeights {
    /// QKV projection weight in Q4K format (combined, legacy)
    pub qkv_weight: Option<Vec<u8>>,
    /// Q projection weight in Q4K format (PMAT-103: separate for fused kernel)
    pub attn_q_weight: Option<Vec<u8>>,
    /// K projection weight in Q4K format (PMAT-103: separate for fused kernel)
    pub attn_k_weight: Option<Vec<u8>>,
    /// V projection weight in Q4K/Q6K format (PMAT-103: separate for fused kernel)
    pub attn_v_weight: Option<Vec<u8>>,
    /// V projection weight in Q6K format (when Q4K not available)
    pub attn_v_weight_q6k: Option<Vec<u8>>,
    /// Attention output projection in Q4K format
    pub attn_output_weight: Option<Vec<u8>>,
    /// FFN gate weight in Q4K format (for SwiGLU)
    pub ffn_gate_weight: Option<Vec<u8>>,
    /// FFN up projection in Q4K format
    pub ffn_up_weight: Option<Vec<u8>>,
    /// FFN down projection in Q4K format
    pub ffn_down_weight: Option<Vec<u8>>,
    /// FFN down projection in Q6K format (when Q4K not available)
    pub ffn_down_weight_q6k: Option<Vec<u8>>,
    /// FFN up projection in Q6K format (when Q4K not available)
    pub ffn_up_weight_q6k: Option<Vec<u8>>,
}

// ============================================================================
// Tests for APR Transformer Configuration (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AprKVCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_kv_cache_new() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), config.context_length);
        assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
        assert_eq!(cache.head_dim(), config.hidden_dim / config.num_heads);
    }

    #[test]
    fn test_apr_kv_cache_append_and_get() {
        let config = AprTransformerConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            hidden_dim: 64,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        // Append to layer 0
        let k = vec![1.0f32; kv_size];
        let v = vec![2.0f32; kv_size];
        cache.append(0, &k, &v);
        cache.advance(); // F-REGR-231: explicit advance required

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        // Get from layer 0
        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), kv_size);
        assert_eq!(v_out.len(), kv_size);
        assert!((k_out[0] - 1.0).abs() < 0.001);
        assert!((v_out[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_apr_kv_cache_multiple_positions() {
        let config = AprTransformerConfig {
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            hidden_dim: 32,
            context_length: 64,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        // Append 3 positions (num_layers=1, so layer 0 is last layer - auto-advances)
        for i in 0..3 {
            let k = vec![(i + 1) as f32; kv_size];
            let v = vec![(i + 10) as f32; kv_size];
            cache.append(0, &k, &v);
            // No advance() needed - append() auto-advances on last layer
        }

        assert_eq!(cache.len(), 3);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), 3 * kv_size);
        assert_eq!(v_out.len(), 3 * kv_size);
    }

    #[test]
    fn test_apr_kv_cache_clear() {
        let config = AprTransformerConfig::default();
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        cache.append(0, &vec![1.0; kv_size], &vec![2.0; kv_size]);
        cache.advance(); // F-REGR-231: explicit advance required
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_apr_kv_cache_debug() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("AprKVCache"));
    }

    #[test]
    fn test_apr_kv_cache_clone() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        let cloned = cache.clone();
        assert_eq!(cloned.len(), cache.len());
        assert_eq!(cloned.capacity(), cache.capacity());
    }

    // -------------------------------------------------------------------------
    // GenerateConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_config_default() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 1.0).abs() < 0.001);
        assert!((config.top_p - 0.9).abs() < 0.001);
        assert_eq!(config.top_k, 0);
        assert!((config.repetition_penalty - 1.0).abs() < 0.001);
        assert!(!config.trace);
    }

    #[test]
    fn test_generate_config_debug() {
        let config = GenerateConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GenerateConfig"));
    }

    #[test]
    fn test_generate_config_clone() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.1,
            trace: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_tokens, 100);
        assert!((cloned.temperature - 0.8).abs() < 0.001);
        assert!(cloned.trace);
    }

    // -------------------------------------------------------------------------
    // AprTransformerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_transformer_config_default() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 2048);
        assert_eq!(config.context_length, 2048);
        assert!((config.rope_theta - 10000.0).abs() < 0.001);
        assert!((config.eps - 1e-5).abs() < 1e-7);
    }

    #[test]
    fn test_apr_transformer_config_debug() {
        let config = AprTransformerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AprTransformerConfig"));
    }

    #[test]
    fn test_apr_transformer_config_clone() {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.architecture, "llama");
        assert_eq!(cloned.hidden_dim, 4096);
    }

    #[test]
    fn test_apr_transformer_config_eq() {
        let config1 = AprTransformerConfig::default();
        let config2 = AprTransformerConfig::default();
        assert_eq!(config1, config2);

        let config3 = AprTransformerConfig {
            hidden_dim: 1024,
            ..Default::default()
        };
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_apr_transformer_config_serialization() {
        let config = AprTransformerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: AprTransformerConfig =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, deserialized);
    }

    // -------------------------------------------------------------------------
    // AprTransformerLayer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_transformer_layer_empty() {
        let layer = AprTransformerLayer::empty(512, 2048);
        assert_eq!(layer.attn_norm_weight.len(), 512);
        assert_eq!(layer.qkv_weight.len(), 512 * 3 * 512);
        assert_eq!(layer.attn_output_weight.len(), 512 * 512);
        assert_eq!(layer.ffn_up_weight.len(), 512 * 2048);
        assert_eq!(layer.ffn_down_weight.len(), 2048 * 512);
        assert!(layer.attn_norm_bias.is_none());
        assert!(layer.ffn_gate_weight.is_none());
    }

    #[test]
    fn test_apr_transformer_layer_empty_gqa() {
        // GQA: 8 query heads, 2 kv heads, head_dim = 64
        let hidden_dim = 512; // 8 heads * 64 head_dim
        let num_heads = 8;
        let num_kv_heads = 2;
        let intermediate_dim = 2048;

        let layer = AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

        let head_dim = hidden_dim / num_heads; // 64
        let kv_dim = num_kv_heads * head_dim; // 2 * 64 = 128
        let qkv_out_dim = hidden_dim + 2 * kv_dim; // 512 + 256 = 768

        assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
        assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
    }

    #[test]
    fn test_apr_transformer_layer_num_parameters() {
        let layer = AprTransformerLayer::empty(64, 128);
        let params = layer.num_parameters();

        // Count expected parameters
        let expected = 64  // attn_norm_weight
            + 64 * 3 * 64  // qkv_weight
            + 64 * 64      // attn_output_weight
            + 64 * 128     // ffn_up_weight
            + 128 * 64;    // ffn_down_weight

        assert_eq!(params, expected);
    }

    #[test]
    fn test_apr_transformer_layer_num_parameters_with_bias() {
        let mut layer = AprTransformerLayer::empty(64, 128);
        layer.attn_norm_bias = Some(vec![0.0; 64]);
        layer.qkv_bias = Some(vec![0.0; 3 * 64]);
        layer.ffn_up_bias = Some(vec![0.0; 128]);

        let params_without = AprTransformerLayer::empty(64, 128).num_parameters();
        let params_with = layer.num_parameters();

        assert_eq!(params_with, params_without + 64 + 3 * 64 + 128);
    }

    #[test]
    fn test_apr_transformer_layer_debug() {
        let layer = AprTransformerLayer::empty(64, 128);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("AprTransformerLayer"));
    }

    #[test]
    fn test_apr_transformer_layer_clone() {
        let layer = AprTransformerLayer::empty(64, 128);
        let cloned = layer.clone();
        assert_eq!(cloned.attn_norm_weight.len(), layer.attn_norm_weight.len());
    }

    #[test]
    fn test_apr_transformer_layer_serialization() {
        let layer = AprTransformerLayer::empty(32, 64);
        let json = serde_json::to_string(&layer).expect("serialize");
        let deserialized: AprTransformerLayer =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.attn_norm_weight.len(), 32);
    }

    // -------------------------------------------------------------------------
    // Q4KLayerWeights Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_q4k_layer_weights_default() {
        let weights = Q4KLayerWeights::default();
        assert!(weights.qkv_weight.is_none());
        assert!(weights.attn_q_weight.is_none());
        assert!(weights.attn_k_weight.is_none());
        assert!(weights.attn_v_weight.is_none());
        assert!(weights.ffn_gate_weight.is_none());
        assert!(weights.ffn_up_weight.is_none());
        assert!(weights.ffn_down_weight.is_none());
    }

    #[test]
    fn test_q4k_layer_weights_debug() {
        let weights = Q4KLayerWeights::default();
        let debug_str = format!("{:?}", weights);
        assert!(debug_str.contains("Q4KLayerWeights"));
    }

    #[test]
    fn test_q4k_layer_weights_clone() {
        let mut weights = Q4KLayerWeights::default();
        weights.qkv_weight = Some(vec![1, 2, 3, 4]);
        let cloned = weights.clone();
        assert_eq!(cloned.qkv_weight, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_q4k_layer_weights_serialization() {
        let mut weights = Q4KLayerWeights::default();
        weights.attn_q_weight = Some(vec![0x12, 0x34]);
        let json = serde_json::to_string(&weights).expect("serialize");
        let deserialized: Q4KLayerWeights =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.attn_q_weight, Some(vec![0x12, 0x34]));
    }
}
