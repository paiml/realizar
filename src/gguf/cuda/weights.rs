//! Weight management methods for CUDA-accelerated inference
//!
//! This module contains weight upload and caching implementations:
//! - `pre_cache_weights_for_batch`: Pre-cache weights for batched forward pass
//! - `preload_weights_gpu`: Upload all layer weights to GPU with indexed lookup
//! - `clear_decode_graph`: Clear CUDA graph state
//! - `supports_gpu_resident`: Check if model supports GPU-resident path

use crate::error::{RealizarError, Result};
use super::{OwnedQuantizedModelCuda, OwnedQKVWeights};
use super::verbose;

impl OwnedQuantizedModelCuda {
    /// PAR-103: Pre-cache all weights for batched forward pass.
    ///
    /// This loads all layer weights into GPU memory with the naming convention
    /// expected by `forward_batch_cuda_native`. Required before using batch mode.
    ///
    /// # Returns
    ///
    /// Total bytes of weights uploaded to GPU.
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails.
    pub fn pre_cache_weights_for_batch(&mut self) -> Result<usize> {
        let mut total_bytes = 0usize;
        let num_layers = self.model.layers.len();

        eprintln!(
            "[PAR-103] Pre-caching {} layer weights for batch mode...",
            num_layers
        );

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layer.{}", layer_idx);

            // Cache QKV weights
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    let q_name = format!("{}.attn_q.weight", prefix);
                    let k_name = format!("{}.attn_k.weight", prefix);
                    let v_name = format!("{}.attn_v.weight", prefix);

                    total_bytes += self
                        .executor
                        .load_quantized_weights(&q_name, &q.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache Q weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&k_name, &k.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache K weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&v_name, &v.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache V weights: {}", e),
                        })?;
                },
                OwnedQKVWeights::Fused(qkv) => {
                    let qkv_name = format!("{}.attn_qkv.weight", prefix);
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&qkv_name, &qkv.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache QKV weights: {}", e),
                        })?;
                },
            }

            // Cache O projection
            let o_name = format!("{}.attn_output.weight", prefix);
            total_bytes += self
                .executor
                .load_quantized_weights(&o_name, &layer.attn_output_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache O weights: {}", e),
                })?;

            // Cache FFN weights (ffn_gate is optional - only SwiGLU models have it)
            let ffn_up_name = format!("{}.ffn_up.weight", prefix);
            let ffn_down_name = format!("{}.ffn_down.weight", prefix);

            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_up_name, &layer.ffn_up_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN up weights: {}", e),
                })?;
            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_down_name, &layer.ffn_down_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN down weights: {}", e),
                })?;

            // FFN gate is optional (SwiGLU models like LLaMA/Qwen)
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let ffn_gate_name = format!("{}.ffn_gate.weight", prefix);
                total_bytes += self
                    .executor
                    .load_quantized_weights(&ffn_gate_name, &gate_weight.data)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "pre_cache_weights_for_batch".to_string(),
                        reason: format!("Failed to cache FFN gate weights: {}", e),
                    })?;
            }
        }

        // Cache LM head
        let lm_head_name = "output.weight".to_string();
        total_bytes += self
            .executor
            .load_quantized_weights(&lm_head_name, &self.model.lm_head_weight.data)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "pre_cache_weights_for_batch".to_string(),
                reason: format!("Failed to cache LM head weights: {}", e),
            })?;

        let total_mb = total_bytes / (1024 * 1024);
        eprintln!(
            "[PAR-103] Pre-cached {} MB of weights for batch mode",
            total_mb
        );
        Ok(total_bytes)
    }

    /// PAR-023: Pre-upload all layer weights to GPU with naming convention for
    /// GPU-resident transformer layer.
    ///
    /// This method uploads quantized weights using names expected by
    /// `CudaExecutor::transformer_layer_gpu`:
    /// - `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`
    /// - `blk.{i}.attn_output.weight`
    /// - `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails or model uses fused QKV (phi-2 style).
    pub fn preload_weights_gpu(&mut self) -> Result<usize> {
        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        let mut total_bytes = 0usize;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            // Upload Q, K, V weights (requires separate format for GPU-resident path)
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    // Q projection - PAR-058: pass qtype for mixed-quant models (e.g., Q5_0 in Qwen 0.5B)
                    let q_name = format!("{}.attn_q.weight", prefix);
                    if !self.executor.has_quantized_weights(&q_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&q_name, &q.data, q.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload Q weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }

                    // K projection - PAR-058: pass qtype for mixed-quant models
                    let k_name = format!("{}.attn_k.weight", prefix);
                    if !self.executor.has_quantized_weights(&k_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&k_name, &k.data, k.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload K weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }

                    // V projection - PAR-058: pass qtype for mixed-quant models
                    let v_name = format!("{}.attn_v.weight", prefix);
                    if !self.executor.has_quantized_weights(&v_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&v_name, &v.data, v.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload V weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }
                },
                OwnedQKVWeights::Fused(_) => {
                    // Fused QKV not yet supported for GPU-resident path
                    // Fall back to standard forward pass for phi-2 style models
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Layer {} uses fused QKV (phi-2 style), GPU-resident path requires separate Q/K/V",
                            layer_idx
                        ),
                    });
                },
            }

            // Output projection - PAR-058: pass qtype for mixed-quant models
            let o_name = format!("{}.attn_output.weight", prefix);
            if !self.executor.has_quantized_weights(&o_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &o_name,
                        &layer.attn_output_weight.data,
                        layer.attn_output_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload O weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }

            // FFN gate (SwiGLU models) - PAR-058: pass qtype
            if let Some(ref gate) = layer.ffn_gate_weight {
                let gate_name = format!("{}.ffn_gate.weight", prefix);
                if !self.executor.has_quantized_weights(&gate_name) {
                    total_bytes += self
                        .executor
                        .load_quantized_weights_with_type(&gate_name, &gate.data, gate.qtype)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "preload_weights_gpu".to_string(),
                            reason: format!(
                                "Failed to upload gate weights for layer {}: {}",
                                layer_idx, e
                            ),
                        })?;
                }
            }

            // FFN up - PAR-058: pass qtype
            let up_name = format!("{}.ffn_up.weight", prefix);
            if !self.executor.has_quantized_weights(&up_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &up_name,
                        &layer.ffn_up_weight.data,
                        layer.ffn_up_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload up weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }

            // FFN down - PAR-058: pass qtype
            let down_name = format!("{}.ffn_down.weight", prefix);
            if !self.executor.has_quantized_weights(&down_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &down_name,
                        &layer.ffn_down_weight.data,
                        layer.ffn_down_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload down weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }
        }

        // Also upload LM head weights
        let lm_head_name = "output.weight".to_string();
        if !self.executor.has_quantized_weights(&lm_head_name) {
            // PAR-060-DEBUG: Print first bytes of LM head weight for verification
            let lm_data = &self.model.lm_head_weight.data;
            if verbose() {
                eprintln!(
                    "[PAR-060-DEBUG] LM head weight: len={}, first 20 bytes: {:?}",
                    lm_data.len(),
                    &lm_data[..20.min(lm_data.len())]
                );
                eprintln!(
                    "[PAR-060-DEBUG] LM head dims: in_dim={}, out_dim={}",
                    self.model.lm_head_weight.in_dim, self.model.lm_head_weight.out_dim
                );
            }
            total_bytes += self
                .executor
                .load_quantized_weights_with_type(
                    &lm_head_name,
                    lm_data,
                    self.model.lm_head_weight.qtype,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("Failed to upload LM head weights: {}", e),
                })?;
        }

        // PAR-023: Pre-cache RMSNorm weights for all layers
        let num_layers = self.model.layers.len();
        let attn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| l.attn_norm_weight.as_slice())
            .collect();
        let ffn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.ffn_norm_weight
                    .as_ref()
                    .map_or(l.attn_norm_weight.as_slice(), |w| w.as_slice())
            })
            .collect();

        total_bytes += self
            .executor
            .preload_rmsnorm_weights(num_layers, &attn_norms, &ffn_norms)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload RMSNorm weights: {}", e),
            })?;

        // PAR-023: Pre-cache output norm (final layer norm) weight
        // This enables fully GPU-resident forward pass including output norm + LM head
        total_bytes += self
            .executor
            .preload_output_norm(&self.model.output_norm_weight)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload output norm weights: {}", e),
            })?;

        // BIAS-FIX: Pre-cache QKV bias vectors for all layers
        // Qwen2.5 models have QKV bias that must be added after GEMV
        let q_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    // Q bias is first q_dim elements
                    let q_dim = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, .. } => q.out_dim,
                        OwnedQKVWeights::Fused(w) => w.out_dim / 3,
                    };
                    &b[..q_dim]
                })
            })
            .collect();
        let k_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let (q_dim, k_dim) = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, k, .. } => (q.out_dim, k.out_dim),
                        OwnedQKVWeights::Fused(w) => {
                            let dim = w.out_dim / 3;
                            (dim, dim)
                        },
                    };
                    &b[q_dim..q_dim + k_dim]
                })
            })
            .collect();
        let v_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let (q_dim, k_dim, v_dim) = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
                        OwnedQKVWeights::Fused(w) => {
                            let dim = w.out_dim / 3;
                            (dim, dim, dim)
                        },
                    };
                    &b[q_dim + k_dim..q_dim + k_dim + v_dim]
                })
            })
            .collect();

        total_bytes += self
            .executor
            .preload_qkv_bias(num_layers, &q_biases, &k_biases, &v_biases)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload QKV bias: {}", e),
            })?;

        // PAR-064-FIX: Pre-cache LM head bias (output.bias) for models that have it
        // Without this bias, GPU inference produces incorrect token predictions
        total_bytes += self
            .executor
            .preload_lm_head_bias(self.model.lm_head_bias.as_deref())
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload LM head bias: {}", e),
            })?;

        // PAR-043: Build indexed weight lookup table for O(1) access during decode
        // This eliminates ~10ms constant CPU overhead per token from string formatting + HashMap lookups
        // PAR-107: Skip if already indexed to preserve CUDA graph (graph captures buffer addresses)
        if !self.executor.has_indexed_weights() {
            self.executor
                .build_indexed_weights(num_layers, |i| format!("blk.{}", i))
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-043: Failed to build indexed weights: {}", e),
                })?;
        }

        // PAR-044: Initialize workspace buffers for zero-allocation forward pass
        // This eliminates ~288 buffer allocations per token
        // PAR-107: Skip if already initialized to preserve CUDA graph (graph captures buffer addresses)
        // ROOT CAUSE FIX: Reallocating workspace invalidates graph since addresses change
        if !self.executor.has_workspace() {
            self.executor
                .init_workspace(
                    self.model.config.hidden_dim,
                    self.model.config.intermediate_dim,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-044: Failed to initialize workspace: {}", e),
                })?;
        }

        Ok(total_bytes)
    }

    /// Clear decode graph and related state
    ///
    /// Call this before starting a new generation session to ensure
    /// the graph is recaptured with fresh state.
    pub fn clear_decode_graph(&mut self) {
        self.executor.clear_decode_graph();
    }

    /// PAR-023: Check if model supports GPU-resident forward pass
    ///
    /// GPU-resident path requires:
    /// - Separate Q/K/V weights (not fused)
    /// - SwiGLU activation (ffn_gate_weight present)
    /// - RMSNorm (LLaMA-style architecture)
    #[must_use]
    pub fn supports_gpu_resident(&self) -> bool {
        // Check first layer for architecture detection
        if let Some(layer) = self.model.layers.first() {
            // Must have separate Q/K/V
            let has_separate_qkv = matches!(layer.qkv_weight, OwnedQKVWeights::Separate { .. });
            // Must have SwiGLU (gate weight present)
            let has_swiglu = layer.ffn_gate_weight.is_some();
            // Must have FFN norm (RMSNorm for pre-FFN)
            let has_ffn_norm = layer.ffn_norm_weight.is_some();

            has_separate_qkv && has_swiglu && has_ffn_norm
        } else {
            false
        }
    }
}
