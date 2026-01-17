//! CUDA Inference Engine
//!
//! Provides CUDA-accelerated transformer inference for NVIDIA GPUs.
//! This module wraps `OwnedQuantizedModelCuda` with a clean interface.

use crate::error::Result;
use crate::gguf::{GGUFConfig, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};

/// CUDA inference engine for GPU-accelerated inference
///
/// This provides the highest performance inference path using CUDA.
/// Supports both full CUDA inference and GPU-resident generation.
///
/// # Example
///
/// ```ignore
/// use realizar::engine::CudaInferenceEngine;
/// use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
///
/// let mapped = MappedGGUFModel::from_path("model.gguf")?;
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let engine = CudaInferenceEngine::new(model, 0)?; // Device 0
///
/// let config = QuantizedGenerateConfig::default().with_max_tokens(100);
/// let tokens = engine.generate_gpu_resident(&[1, 2, 3], &config)?;
/// ```
pub struct CudaInferenceEngine {
    model: OwnedQuantizedModelCuda,
}

impl CudaInferenceEngine {
    /// Create a new CUDA inference engine
    ///
    /// # Arguments
    /// * `model` - Owned quantized model to wrap
    /// * `device_ordinal` - CUDA device index (0 for first GPU)
    ///
    /// # Errors
    /// Returns error if CUDA initialization fails
    pub fn new(model: OwnedQuantizedModel, device_ordinal: i32) -> Result<Self> {
        let cuda_model = OwnedQuantizedModelCuda::new(model, device_ordinal)?;
        Ok(Self { model: cuda_model })
    }

    /// Create with custom maximum sequence length
    ///
    /// # Errors
    /// Returns error if CUDA initialization fails
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        let cuda_model =
            OwnedQuantizedModelCuda::with_max_seq_len(model, device_ordinal, max_seq_len)?;
        Ok(Self { model: cuda_model })
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn is_available() -> bool {
        OwnedQuantizedModelCuda::is_available()
    }

    /// Get number of available CUDA devices
    #[must_use]
    pub fn num_devices() -> usize {
        OwnedQuantizedModelCuda::num_devices()
    }

    /// Get device name
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.model.device_name()
    }

    /// Get VRAM info (free, total) in bytes
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.model.memory_info()
    }

    /// Get VRAM in MB
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        self.model.vram_mb()
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        self.model.model().config()
    }

    /// Generate tokens using full CUDA acceleration with KV cache
    ///
    /// This is the recommended method for GPU inference - it keeps
    /// all computation on the GPU and minimizes CPU-GPU transfers.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if CUDA operations fail
    pub fn generate_full_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        self.model.generate_full_cuda_with_cache(prompt, config)
    }

    /// Generate using GPU-resident mode
    ///
    /// Keeps model weights resident on GPU for maximum performance.
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_gpu_resident(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        self.model.generate_gpu_resident(prompt, config)
    }

    /// Get the underlying CUDA model (for advanced use cases)
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModelCuda {
        &self.model
    }

    /// Get mutable access to the underlying CUDA model
    pub fn model_mut(&mut self) -> &mut OwnedQuantizedModelCuda {
        &mut self.model
    }

    /// Enable profiling
    pub fn enable_profiling(&mut self) {
        self.model.enable_profiling();
    }

    /// Disable profiling
    pub fn disable_profiling(&mut self) {
        self.model.disable_profiling();
    }

    /// Get profiler summary
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.model.profiler_summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability_check() {
        // This should not panic regardless of CUDA availability
        let _ = CudaInferenceEngine::is_available();
        let _ = CudaInferenceEngine::num_devices();
    }
}
