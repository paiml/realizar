//! Batch Inference Engine
//!
//! Provides batched inference for API serving with GPU acceleration.
//! This module wraps `OwnedQuantizedModelCachedSync` for concurrent batch processing.

use crate::error::Result;
use crate::gguf::{
    DispatchMetrics, GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
};
use std::sync::Arc;

/// Batch inference engine for API serving
///
/// This provides high-throughput inference by batching multiple requests
/// and leveraging GPU parallelism where available.
///
/// # Example
///
/// ```ignore
/// use realizar::engine::BatchInferenceEngine;
/// use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
///
/// let mapped = MappedGGUFModel::from_path("model.gguf")?;
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let engine = BatchInferenceEngine::new(model);
///
/// let prompts = vec![vec![1, 2, 3], vec![4, 5, 6]];
/// let config = QuantizedGenerateConfig::default();
/// let results = engine.batch_generate(&prompts, &config)?;
/// ```
pub struct BatchInferenceEngine {
    model: OwnedQuantizedModelCachedSync,
}

impl BatchInferenceEngine {
    /// Create a new batch inference engine
    #[must_use]
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model: OwnedQuantizedModelCachedSync::new(model),
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        self.model.model().config()
    }

    /// Generate tokens for a batch of prompts using GPU acceleration
    ///
    /// # Arguments
    /// * `prompts` - Batch of prompt token sequences
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Vector of generated token sequences (one per prompt)
    ///
    /// # Errors
    /// Returns error if batch generation fails
    pub fn batch_generate(
        &self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        self.model.batch_generate_gpu(prompts, config)
    }

    /// Generate tokens with KV caching
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        self.model.generate_with_cache(prompt, config)
    }

    /// Generate with adaptive attention (auto-selects CPU/GPU)
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        self.model
            .generate_with_cache_adaptive(prompt, config, metrics)
    }

    /// Warmup GPU cache for better first-request latency
    ///
    /// # Returns
    /// Tuple of (layers_cached, bytes_used)
    ///
    /// # Errors
    /// Returns error if warmup fails
    pub fn warmup_gpu_cache(&self) -> Result<(usize, usize)> {
        self.model.warmup_gpu_cache()
    }

    /// Check if GPU cache is warm
    #[must_use]
    pub fn is_gpu_cache_warm(&self) -> bool {
        self.model.is_gpu_cache_warm()
    }

    /// Get GPU cache memory usage in bytes
    #[must_use]
    pub fn gpu_cache_memory(&self) -> usize {
        self.model.gpu_cache_memory()
    }

    /// Get the underlying model (for advanced use cases)
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModelCachedSync {
        &self.model
    }

    /// Create a KV cache for the given maximum sequence length
    #[must_use]
    pub fn create_cache(&self, max_seq_len: usize) -> OwnedQuantizedKVCache {
        OwnedQuantizedKVCache::from_config(self.config(), max_seq_len)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_engine_module_exists() {
        // Basic module existence test
        assert!(true);
    }
}
