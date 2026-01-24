//! Runtime types for GGUF model inference
//!
//! This module contains types used during inference runtime:
//!
//! - `QuantizedGenerateConfig`: Configuration for text generation
//! - `OwnedQuantizedKVCache`: KV cache for incremental decoding
//!
//! These are "leaf nodes" in the dependency graph - they don't depend
//! on other complex types, making them easy to extract.

use super::config::GGUFConfig;

// ============================================================================
// QuantizedGenerateConfig - Generation parameters
// ============================================================================

/// Configuration for quantized generation
///
/// Per benchmark-model-runners-spec.md "What's Remaining" item 1:
/// End-to-end Q4_K inference with generation config.
#[derive(Debug, Clone)]
pub struct QuantizedGenerateConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    pub top_k: usize,
    /// Stop token IDs
    pub stop_tokens: Vec<u32>,
}

impl Default for QuantizedGenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }
}

impl QuantizedGenerateConfig {
    /// Create config for deterministic (greedy) generation
    #[must_use]
    pub fn deterministic(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }
}

// ============================================================================
// OwnedQuantizedKVCache - KV cache for incremental decoding
// ============================================================================

/// KV Cache for OwnedQuantizedModel incremental decoding (IMP-101c)
///
/// Stores Key and Value projections for all layers to enable O(n) per-token
/// decoding instead of O(n²). Reference: Spec Section 5.4 "Continuous Flow".
///
/// Memory layout: [num_layers, seq_len, hidden_dim]
#[derive(Debug, Clone)]
pub struct OwnedQuantizedKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Hidden dimension (stored for future use)
    _hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length (tokens processed)
    seq_len: usize,
    /// Key cache: [num_layers][seq_len][hidden_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache: [num_layers][seq_len][hidden_dim]
    v_cache: Vec<Vec<f32>>,
}

/// PARITY-096: Default impl for std::mem::take optimization in batch_generate_gpu
impl Default for OwnedQuantizedKVCache {
    fn default() -> Self {
        Self {
            num_layers: 0,
            _hidden_dim: 0,
            max_seq_len: 0,
            seq_len: 0,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
        }
    }
}

impl OwnedQuantizedKVCache {
    /// Create a new KV cache for the given model configuration
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension (num_heads * head_dim)
    /// * `max_seq_len` - Maximum sequence length to cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_layers,
            _hidden_dim: hidden_dim,
            max_seq_len,
            seq_len: 0,
            k_cache: vec![Vec::with_capacity(max_seq_len * hidden_dim); num_layers],
            v_cache: vec![Vec::with_capacity(max_seq_len * hidden_dim); num_layers],
        }
    }

    /// Create cache from model configuration
    #[must_use]
    pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self {
        Self::new(config.num_layers, config.hidden_dim, max_seq_len)
    }

    /// Append K and V vectors for a single position to a layer's cache
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k` - Key vector [hidden_dim]
    /// * `v` - Value vector [hidden_dim]
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer < self.num_layers && self.seq_len < self.max_seq_len {
            self.k_cache[layer].extend_from_slice(k);
            self.v_cache[layer].extend_from_slice(v);
        }
    }

    /// Advance the sequence position after processing a token
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// PAR-097: Append multiple K/V entries to a layer's cache (for speculative decode)
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k_all` - Key vectors for batch_size positions [batch_size × kv_dim]
    /// * `v_all` - Value vectors for batch_size positions [batch_size × kv_dim]
    pub fn append_kv(&mut self, layer: usize, k_all: &[f32], v_all: &[f32]) {
        if layer < self.num_layers {
            self.k_cache[layer].extend_from_slice(k_all);
            self.v_cache[layer].extend_from_slice(v_all);
        }
    }

    /// PAR-097: Advance sequence position by n tokens (for speculative decode)
    pub fn advance_by(&mut self, n: usize) {
        self.seq_len = (self.seq_len + n).min(self.max_seq_len);
    }

    /// PAR-098: Rollback cache to a previous position (for speculative decode rejection)
    ///
    /// When draft tokens are rejected, we need to remove their K/V entries.
    /// This truncates each layer's cache to keep only the first `new_len` positions.
    ///
    /// # Arguments
    /// * `new_len` - The new sequence length (must be <= current length)
    /// * `kv_dim` - The dimension of each K/V entry (num_kv_heads * head_dim)
    pub fn rollback_to(&mut self, new_len: usize, kv_dim: usize) {
        if new_len >= self.seq_len {
            return; // Nothing to rollback
        }
        let target_size = new_len * kv_dim;
        for layer_k in &mut self.k_cache {
            layer_k.truncate(target_size);
        }
        for layer_v in &mut self.v_cache {
            layer_v.truncate(target_size);
        }
        self.seq_len = new_len;
    }

    /// PAR-098: Get a snapshot of current cache lengths for rollback
    #[must_use]
    pub fn snapshot_len(&self) -> usize {
        self.seq_len
    }

    /// Get cached keys for a layer
    ///
    /// Returns slice of [seq_len, hidden_dim]
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        if layer < self.num_layers {
            &self.k_cache[layer]
        } else {
            &[]
        }
    }

    /// Get cached values for a layer
    ///
    /// Returns slice of [seq_len, hidden_dim]
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        if layer < self.num_layers {
            &self.v_cache[layer]
        } else {
            &[]
        }
    }

    /// Current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer_k in &mut self.k_cache {
            layer_k.clear();
        }
        for layer_v in &mut self.v_cache {
            layer_v.clear();
        }
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_config_default() {
        let config = QuantizedGenerateConfig::default();
        assert_eq!(config.max_tokens, 64);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generate_config_deterministic() {
        let config = QuantizedGenerateConfig::deterministic(128);
        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = OwnedQuantizedKVCache::new(4, 256, 512);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 512);
    }

    #[test]
    fn test_kv_cache_append_advance() {
        let mut cache = OwnedQuantizedKVCache::new(2, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(0), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = OwnedQuantizedKVCache::new(2, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        cache.reset();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_k(0).is_empty());
    }

    #[test]
    fn test_kv_cache_rollback() {
        let mut cache = OwnedQuantizedKVCache::new(1, 2, 10);

        // Append 3 tokens
        for i in 0..3 {
            let k = vec![i as f32, i as f32 + 0.5];
            let v = vec![i as f32 + 1.0, i as f32 + 1.5];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get_k(0).len(), 6); // 3 tokens * 2 dims

        // Rollback to position 1
        cache.rollback_to(1, 2);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0).len(), 2); // 1 token * 2 dims
    }

    #[test]
    fn test_kv_cache_from_config() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let cache = OwnedQuantizedKVCache::from_config(&config, 512);
        assert_eq!(cache.max_len(), 512);
    }

    #[test]
    fn test_kv_cache_default() {
        let cache = OwnedQuantizedKVCache::default();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 0);
        assert!(cache.is_empty());
    }
}
