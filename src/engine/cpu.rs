//! CPU Inference Engine
//!
//! Provides CPU-based transformer inference with KV caching.
//! This is the primary inference path for non-GPU deployments.

use crate::error::Result;
use crate::gguf::{GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModel};

/// CPU inference engine wrapping an owned quantized model
///
/// This provides a clean interface for CPU-based inference,
/// separating the inference logic from the GGUF parsing code.
///
/// # Example
///
/// ```ignore
/// use realizar::engine::CpuInferenceEngine;
/// use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedKVCache};
///
/// let mapped = MappedGGUFModel::from_path("model.gguf")?;
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let engine = CpuInferenceEngine::new(model);
///
/// let mut cache = engine.create_cache(512);
/// let logits = engine.forward(token_id, &mut cache, position)?;
/// ```
pub struct CpuInferenceEngine {
    model: OwnedQuantizedModel,
}

impl CpuInferenceEngine {
    /// Create a new CPU inference engine from an owned quantized model
    #[must_use]
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self { model }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        self.model.config()
    }

    /// Create a KV cache for the given maximum sequence length
    #[must_use]
    pub fn create_cache(&self, max_seq_len: usize) -> OwnedQuantizedKVCache {
        OwnedQuantizedKVCache::from_config(self.config(), max_seq_len)
    }

    /// Forward pass with KV caching for efficient autoregressive decoding
    ///
    /// This method handles both LLaMA-style (RMSNorm, SwiGLU, GQA) and
    /// phi-2 style (LayerNorm, GELU, MHA) architectures.
    ///
    /// Uses O(n) per-token cost instead of O(n²) by caching K/V.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for all layers
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        self.model.forward_cached(token_id, cache, position)
    }

    /// Get token embedding for a single token
    #[must_use]
    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        self.model.embed(&[token_id])
    }

    /// Get the underlying model (for advanced use cases)
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// Take ownership of the underlying model
    #[must_use]
    pub fn into_model(self) -> OwnedQuantizedModel {
        self.model
    }

    /// Perform greedy argmax over logits
    #[must_use]
    pub fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }

    /// Top-k sampling with temperature
    #[must_use]
    pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        OwnedQuantizedModel::sample_topk(logits, temperature, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_engine_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(CpuInferenceEngine::argmax(&logits), 3);
    }

    #[test]
    fn test_cpu_engine_argmax_empty() {
        let logits: Vec<f32> = vec![];
        assert_eq!(CpuInferenceEngine::argmax(&logits), 0);
    }

    #[test]
    fn test_cpu_engine_argmax_single() {
        let logits = vec![42.0];
        assert_eq!(CpuInferenceEngine::argmax(&logits), 0);
    }
}
