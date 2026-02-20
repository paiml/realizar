//! Persistent GPU weight management for CudaExecutor
//!
//! This module implements:
//! - PARITY-037: FP32 weight loading and caching
//! - PAR-005: Quantized weight cache (Q4_K/Q5_K/Q6_K)
//! - PAR-043: Indexed weight access for O(1) lookup
//! - PAR-058: Mixed-quantization type tracking

use super::*;

impl CudaExecutor {
    // ========================================================================
    // PARITY-037: Persistent GPU Weight Management
    // ========================================================================

    /// Load weights to GPU and cache them for reuse (PARITY-037)
    ///
    /// Weights are stored in GPU memory and persist until explicitly cleared
    /// or the executor is dropped. This eliminates H2D transfer overhead
    /// for repeated forward passes.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the weight tensor (e.g., "layer0.ffn.fc1")
    /// * `weights` - Weight data to upload (row-major)
    ///
    /// # Returns
    ///
    /// Size in bytes of the uploaded weights.
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation or transfer fails.
    pub fn load_weights(&mut self, name: &str, weights: &[f32]) -> Result<usize, GpuError> {
        let buf = GpuBuffer::from_host(&self.context, weights)?;
        let size_bytes = buf.size_bytes();
        self.weight_cache.insert(name.to_string(), buf);
        Ok(size_bytes)
    }

    /// Check if weights are cached on GPU
    #[must_use]
    pub fn has_weights(&self, name: &str) -> bool {
        self.weight_cache.contains_key(name)
    }

    /// Get the number of cached weight tensors
    #[must_use]
    pub fn cached_weight_count(&self) -> usize {
        self.weight_cache.len()
    }

    /// Get total size of cached weights in bytes
    #[must_use]
    pub fn cached_weight_bytes(&self) -> usize {
        self.weight_cache.values().map(GpuBuffer::size_bytes).sum()
    }

    /// Clear all cached weights (releases GPU memory)
    pub fn clear_weights(&mut self) {
        self.weight_cache.clear();
    }

    // ========================================================================
    // PAR-005: Quantized Weight Cache (Q4_K/Q5_K/Q6_K)
    // ========================================================================

    /// Load quantized weights onto GPU for persistent caching
    ///
    /// Uploads raw quantized bytes (Q4_K/Q5_K/Q6_K format) to GPU memory.
    /// These weights are reused for all forward passes, eliminating
    /// the ~50+ CPU→GPU transfers per token.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this weight tensor (e.g., "layer_0.attn_q")
    /// * `data` - Raw quantized weight bytes
    ///
    /// # Returns
    ///
    /// Size in bytes of the uploaded weights.
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation or transfer fails.
    pub fn load_quantized_weights(&mut self, name: &str, data: &[u8]) -> Result<usize, GpuError> {
        // Default to Q4K (type 12) for backwards compatibility
        self.load_quantized_weights_with_type(name, data, 12)
    }

    /// PAR-058: Load quantized weights with explicit quantization type
    ///
    /// Like `load_quantized_weights` but stores the quantization type for later kernel dispatch.
    /// This is needed for mixed-quantization models like Qwen 0.5B where Q/K use Q5_0.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this weight tensor
    /// * `data` - Raw quantized weight bytes
    /// * `qtype` - GGML quantization type (6=Q5_0, 8=Q8_0, 12=Q4K, 13=Q5K, 14=Q6K)
    ///
    /// # Returns
    ///
    /// Size in bytes of the uploaded weights.
    pub fn load_quantized_weights_with_type(
        &mut self,
        name: &str,
        data: &[u8],
        qtype: u32,
    ) -> Result<usize, GpuError> {
        let buf = GpuBuffer::from_host(&self.context, data)?;
        let size_bytes = buf.size_bytes();
        self.quantized_weight_cache.insert(name.to_string(), buf);
        self.quantized_weight_types.insert(name.to_string(), qtype);
        Ok(size_bytes)
    }

    /// PAR-058: Get the quantization type for a cached weight
    ///
    /// Returns the GGML type ID (6=Q5_0, 8=Q8_0, 12=Q4K, 13=Q5K, 14=Q6K).
    /// Returns None if the weight is not cached.
    #[must_use]
    pub fn get_quantized_weight_type(&self, name: &str) -> Option<u32> {
        self.quantized_weight_types.get(name).copied()
    }

    /// Check if quantized weights are cached on GPU
    #[must_use]
    pub fn has_quantized_weights(&self, name: &str) -> bool {
        self.quantized_weight_cache.contains_key(name)
    }

    /// Get raw device pointer for cached quantized weights
    ///
    /// Returns the raw u64 device pointer for the named weight buffer.
    /// Used for debugging and direct kernel invocation.
    ///
    /// # Errors
    ///
    /// Returns error if weight is not cached.
    pub fn get_quantized_weight_ptr(&self, name: &str) -> Result<u64, GpuError> {
        self.quantized_weight_cache
            .get(name)
            .map(trueno_gpu::driver::GpuBuffer::as_ptr)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!("Quantized weight '{}' not cached", name))
            })
    }

    /// Get the number of cached quantized weight tensors
    #[must_use]
    pub fn cached_quantized_weight_count(&self) -> usize {
        self.quantized_weight_cache.len()
    }

    /// Get total size of cached quantized weights in bytes
    #[must_use]
    pub fn cached_quantized_weight_bytes(&self) -> usize {
        self.quantized_weight_cache
            .values()
            .map(GpuBuffer::size_bytes)
            .sum()
    }

    /// Clear all cached quantized weights (releases GPU memory)
    pub fn clear_quantized_weights(&mut self) {
        self.quantized_weight_cache.clear();
    }

    // ========================================================================
    // PAR-043: Indexed Weight Access (eliminate HashMap/string overhead)
    // ========================================================================

    /// Build indexed weight lookup table from loaded caches.
    ///
    /// GH-279: Now takes `ArchConstraints` and validates every layer's weights
    /// against the architecture's required roles. If any required weight is
    /// missing (ptr=0, len=0), returns a descriptive error — never silent garbage.
    ///
    /// MUST be called after all weights are loaded via `load_quantized_weights()` and
    /// `load_rmsnorm_gamma()`. This pre-computes device pointers for O(1) access
    /// during decode, eliminating ~10ms constant overhead per token.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers in the model
    /// * `layer_prefix_fn` - Function to generate layer prefix from index (e.g., `|i| format!("blk.{}", i)`)
    /// * `arch` - Architecture constraints for weight validation (GH-279)
    ///
    /// # Errors
    ///
    /// Returns error if any required weight is not cached, or if architecture
    /// validation fails (missing required weight for the declared architecture).
    pub fn build_indexed_weights<F>(
        &mut self,
        num_layers: usize,
        layer_prefix_fn: F,
        arch: &crate::gguf::ArchConstraints,
    ) -> Result<(), GpuError>
    where
        F: Fn(usize) -> String,
    {
        let mut indexed = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let prefix = layer_prefix_fn(layer_idx);

            // Build weight names matching GGML convention
            let q_name = format!("{}.attn_q.weight", prefix);
            let k_name = format!("{}.attn_k.weight", prefix);
            let v_name = format!("{}.attn_v.weight", prefix);
            let o_name = format!("{}.attn_output.weight", prefix);
            let gate_name = format!("{}.ffn_gate.weight", prefix);
            let up_name = format!("{}.ffn_up.weight", prefix);
            let down_name = format!("{}.ffn_down.weight", prefix);
            let attn_norm_name = format!("{}.attn_norm.gamma", prefix);
            let ffn_norm_name = format!("{}.ffn_norm.gamma", prefix);

            // Get pointers from quantized weight cache
            let get_qweight = |name: &str| -> Result<(u64, usize), GpuError> {
                let buf = self.quantized_weight_cache.get(name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-043: Quantized weight '{}' not cached",
                        name
                    ))
                })?;
                Ok((buf.as_ptr(), buf.size_bytes()))
            };

            // Get pointers from RMSNorm cache
            let get_rmsnorm = |name: &str| -> Result<(u64, usize), GpuError> {
                let buf = self.rmsnorm_cache.get(name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-043: RMSNorm gamma '{}' not cached",
                        name
                    ))
                })?;
                Ok((buf.as_ptr(), buf.len()))
            };

            let (attn_q_ptr, attn_q_len) = get_qweight(&q_name)?;
            let (attn_k_ptr, attn_k_len) = get_qweight(&k_name)?;
            let (attn_v_ptr, attn_v_len) = get_qweight(&v_name)?;
            let (attn_output_ptr, attn_output_len) = get_qweight(&o_name)?;
            let (ffn_gate_ptr, ffn_gate_len) = get_qweight(&gate_name)?;
            let (ffn_up_ptr, ffn_up_len) = get_qweight(&up_name)?;
            let (ffn_down_ptr, ffn_down_len) = get_qweight(&down_name)?;
            let (attn_norm_ptr, attn_norm_len) = get_rmsnorm(&attn_norm_name)?;
            let (ffn_norm_ptr, ffn_norm_len) = get_rmsnorm(&ffn_norm_name)?;

            // PAR-058: Resolve quantization types for all weight tensors
            let attn_q_qtype = self.resolve_qtype(&q_name);
            let attn_k_qtype = self.resolve_qtype(&k_name);
            let attn_v_qtype = self.resolve_qtype(&v_name);
            let attn_output_qtype = self.resolve_qtype(&o_name);
            let ffn_gate_qtype = self.resolve_qtype(&gate_name);
            let ffn_up_qtype = self.resolve_qtype(&up_name);
            let ffn_down_qtype = self.resolve_qtype(&down_name);

            // Log if non-Q4K types detected (for debugging mixed-quant models)
            self.log_mixed_quant_types(
                layer_idx,
                attn_q_qtype,
                attn_k_qtype,
                attn_v_qtype,
                attn_output_qtype,
                ffn_gate_qtype,
                ffn_up_qtype,
                ffn_down_qtype,
            );

            // BIAS-FIX: Get QKV bias pointers from bias_cache (optional - 0/0 if not present)
            let q_bias_name = format!("{}.attn_q.bias", prefix);
            let k_bias_name = format!("{}.attn_k.bias", prefix);
            let v_bias_name = format!("{}.attn_v.bias", prefix);

            let (attn_q_bias_ptr, attn_q_bias_len) = self
                .bias_cache
                .get(&q_bias_name)
                .map_or((0, 0), |b| (b.as_ptr(), b.len()));
            let (attn_k_bias_ptr, attn_k_bias_len) = self
                .bias_cache
                .get(&k_bias_name)
                .map_or((0, 0), |b| (b.as_ptr(), b.len()));
            let (attn_v_bias_ptr, attn_v_bias_len) = self
                .bias_cache
                .get(&v_bias_name)
                .map_or((0, 0), |b| (b.as_ptr(), b.len()));

            // GH-279: QkNorm pointers from rmsnorm_cache (optional - 0/0 if not present)
            let q_norm_name = format!("{}.attn_q_norm.gamma", prefix);
            let k_norm_name = format!("{}.attn_k_norm.gamma", prefix);
            let (attn_q_norm_ptr, attn_q_norm_len) = self
                .rmsnorm_cache
                .get(&q_norm_name)
                .map_or((0, 0), |b| (b.as_ptr(), b.len()));
            let (attn_k_norm_ptr, attn_k_norm_len) = self
                .rmsnorm_cache
                .get(&k_norm_name)
                .map_or((0, 0), |b| (b.as_ptr(), b.len()));

            let raw = IndexedLayerWeights {
                attn_q_ptr,
                attn_q_len,
                attn_q_qtype,
                attn_k_ptr,
                attn_k_len,
                attn_k_qtype,
                attn_v_ptr,
                attn_v_len,
                attn_v_qtype,
                attn_output_ptr,
                attn_output_len,
                attn_output_qtype, // PAR-058: was missing
                ffn_gate_ptr,
                ffn_gate_len,
                ffn_gate_qtype, // PAR-058: was missing
                ffn_up_ptr,
                ffn_up_len,
                ffn_up_qtype, // PAR-058: was missing
                ffn_down_ptr,
                ffn_down_len,
                ffn_down_qtype,
                attn_norm_ptr,
                attn_norm_len,
                ffn_norm_ptr,
                ffn_norm_len,
                // BIAS-FIX: QKV bias pointers
                attn_q_bias_ptr,
                attn_q_bias_len,
                attn_k_bias_ptr,
                attn_k_bias_len,
                attn_v_bias_ptr,
                attn_v_bias_len,
                // GH-279: QkNorm pointers (Qwen3 per-head RMSNorm)
                attn_q_norm_ptr,
                attn_q_norm_len,
                attn_k_norm_ptr,
                attn_k_norm_len,
            };

            // GH-279: Validate that all architecture-required fields are non-zero.
            // This is the Poka-Yoke enforcement point — if a loader forgot to
            // populate a required field, we fail HERE (not during inference).
            use crate::cuda::types::ValidatedLayerWeights;
            let validated = ValidatedLayerWeights::validate(raw, arch, layer_idx)
                .map_err(|e| GpuError::InvalidLaunchConfig(e.to_string()))?;

            indexed.push(validated);
        }

        self.indexed_layer_weights = indexed;
        self.index_output_weights();
        Ok(())
    }

    /// Check if indexed weights have been built
    #[must_use]
    pub fn has_indexed_weights(&self) -> bool {
        !self.indexed_layer_weights.is_empty()
    }

    /// Get validated indexed weights for a specific layer.
    ///
    /// GH-279: Returns `&ValidatedLayerWeights` — all architecture-required fields
    /// are guaranteed non-zero by construction.
    ///
    /// # Panics
    ///
    /// Panics if `layer_idx >= num_layers` or if `build_indexed_weights()` hasn't been called.
    #[must_use]
    pub fn get_indexed_layer(&self, layer_idx: usize) -> &ValidatedLayerWeights {
        &self.indexed_layer_weights[layer_idx]
    }

    /// Clear indexed weights (call before reloading model)
    pub fn clear_indexed_weights(&mut self) {
        self.indexed_layer_weights.clear();
        self.output_norm_ptr = 0;
        self.output_norm_len = 0;
        self.lm_head_ptr = 0;
        self.lm_head_len = 0;
        self.lm_head_qtype = WeightQuantType::Q4K;
        // PAR-064-FIX: Also clear LM head bias pointer
        self.lm_head_bias_ptr = 0;
        self.lm_head_bias_len = 0;
    }

    // ========================================================================
    // PAR-058: Helper methods for quantization type resolution
    // ========================================================================

    /// Resolve the quantization type for a named weight tensor.
    ///
    /// Looks up the GGML type stored during `load_quantized_weights_with_type()`,
    /// converts it to `WeightQuantType`, and defaults to Q4K if not found.
    fn resolve_qtype(&self, name: &str) -> WeightQuantType {
        self.quantized_weight_types
            .get(name)
            .and_then(|&t| WeightQuantType::from_ggml_type(t))
            .unwrap_or(WeightQuantType::Q4K)
    }

    /// Index output norm and LM head pointers for zero-allocation forward pass.
    ///
    /// PAR-054: LM head weight for CUDA graph capture.
    /// PAR-058: Detect LM head quantization type (Q6_K in Qwen 1.5B, not Q4_K).
    fn index_output_weights(&mut self) {
        if let Some(buf) = self.rmsnorm_cache.get("output_norm.gamma") {
            self.output_norm_ptr = buf.as_ptr();
            self.output_norm_len = buf.len();
        }

        if let Some(buf) = self.quantized_weight_cache.get("output.weight") {
            self.lm_head_ptr = buf.as_ptr();
            self.lm_head_len = buf.len();
            self.lm_head_qtype = self.resolve_qtype("output.weight");
            if verbose() {
                eprintln!(
                    "[PAR-058] LM head qtype: {:?}, ptr={:#x}, len={}",
                    self.lm_head_qtype, self.lm_head_ptr, self.lm_head_len
                );
            }
        }
    }

    /// Log non-Q4K quantization types for debugging mixed-quant models (PAR-058).
    ///
    /// Only emits output when `verbose()` is true and at least one weight
    /// uses a quantization type other than Q4K.
    #[allow(clippy::too_many_arguments)]
    fn log_mixed_quant_types(
        &self,
        layer_idx: usize,
        attn_q: WeightQuantType,
        attn_k: WeightQuantType,
        attn_v: WeightQuantType,
        attn_output: WeightQuantType,
        ffn_gate: WeightQuantType,
        ffn_up: WeightQuantType,
        ffn_down: WeightQuantType,
    ) {
        if !verbose() {
            return;
        }
        if attn_q != WeightQuantType::Q4K || attn_k != WeightQuantType::Q4K {
            eprintln!(
                "[PAR-058] Layer {}: Q={:?}, K={:?}, V={:?}",
                layer_idx, attn_q, attn_k, attn_v
            );
        }
        if attn_output != WeightQuantType::Q4K
            || ffn_gate != WeightQuantType::Q4K
            || ffn_up != WeightQuantType::Q4K
            || ffn_down != WeightQuantType::Q4K
        {
            eprintln!(
                "[PAR-058] Layer {}: O={:?}, gate={:?}, up={:?}, down={:?}",
                layer_idx, attn_output, ffn_gate, ffn_up, ffn_down
            );
        }
    }
}

include!("weights_tests.rs");
