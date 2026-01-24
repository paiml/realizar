//! GGUF model types
//!
//! This module contains the core model structures for GGUF inference:
//!
//! - `MappedGGUFModel`: Memory-mapped GGUF file with zero-copy access
//! - `GGUFTransformer`: F32 transformer weights (dequantized from GGUF)
//! - `GGUFTransformerLayer`: Per-layer transformer weights
//! - `OwnedQuantizedModel`: Quantized model with owned weight data
//!
//! ## Design Philosophy
//!
//! Per Wulf & McKee (1995) "Hitting the Memory Wall", memory bandwidth is the
//! bottleneck for LLM inference. These types support:
//! - Zero-copy mmap loading (`MappedGGUFModel`)
//! - Quantized weights for 8x bandwidth reduction (`OwnedQuantizedModel`)
//! - Lazy dequantization during computation

use memmap2::Mmap;

use super::config::GGUFConfig;
use super::quantized::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedTensor};
use super::types::GGUFModel;

// ============================================================================
// MappedGGUFModel - Zero-copy memory-mapped model
// ============================================================================

/// Memory-mapped GGUF model for zero-copy tensor access
///
/// Uses `memmap2` for efficient large model loading without copying
/// entire file contents into heap memory.
///
/// # Example
///
/// ```rust,ignore
/// let model = MappedGGUFModel::from_path("phi-2-q4_k_m.gguf")?;
/// println!("Loaded {} tensors", model.model.tensors.len());
/// ```
pub struct MappedGGUFModel {
    /// Parsed model metadata (header, tensors, etc.)
    pub model: GGUFModel,
    /// Memory-mapped file contents
    pub(crate) mmap: Mmap,
}

// ============================================================================
// GGUFTransformer - F32 transformer weights
// ============================================================================

/// F32 transformer weights loaded from GGUF
///
/// This struct holds dequantized weights in F32 format.
/// Used for reference implementations and debugging.
/// For production inference, use `OwnedQuantizedModel` instead.
pub struct GGUFTransformer {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embedding weights [vocab_size, hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Attention weights per layer
    pub layers: Vec<GGUFTransformerLayer>,
    /// Output norm weight
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head / output projection weight
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional)
    pub lm_head_bias: Option<Vec<f32>>,
}

// ============================================================================
// GGUFTransformerLayer - Per-layer F32 weights
// ============================================================================

/// Weights for a single transformer layer
pub struct GGUFTransformerLayer {
    /// Attention norm weight
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (combined for phi-2, concatenated Q+K+V for llama)
    pub qkv_weight: Vec<f32>,
    /// QKV bias (phi-2 has bias, llama doesn't)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate projection weight (SwiGLU models like llama)
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate projection bias
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (for models with separate FFN normalization)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias
    pub ffn_norm_bias: Option<Vec<f32>>,
}

// ============================================================================
// OwnedQuantizedModel - Quantized model with owned data
// ============================================================================

/// Owned quantized model with all weight data
///
/// IMP-100: This struct owns all quantized weight data, allowing storage
/// in `Arc` without lifetime parameters. Essential for async handlers.
///
/// # Memory Layout
///
/// - Token embedding: F32 for fast lookup
/// - Layer weights: Quantized (Q4_K, Q6_K, etc.)
/// - Output norm: F32 (small)
/// - LM head: Quantized
///
/// # GPU Acceleration
///
/// When the `cuda` feature is enabled, this struct includes a
/// `CudaExecutor` for GPU-accelerated matmul operations.
pub struct OwnedQuantizedModel {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embedding (f32 for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Owned quantized layers
    pub layers: Vec<OwnedQuantizedLayer>,
    /// Output norm weight (f32)
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (owned quantized)
    pub lm_head_weight: OwnedQuantizedTensor,
    /// LM head bias (optional, f32)
    pub lm_head_bias: Option<Vec<f32>>,
    /// PARITY-113: Optional CUDA executor for GPU acceleration
    /// When present, fused_matmul routes to CUDA GEMM kernels
    /// Uses Mutex for thread-safety in async handlers
    #[cfg(feature = "cuda")]
    pub(crate) cuda_executor: Option<std::sync::Mutex<crate::cuda::CudaExecutor>>,
    /// Track CUDA kernel execution count for metrics
    /// Uses AtomicU64 for thread-safe counting
    #[cfg(feature = "cuda")]
    pub(crate) cuda_kernel_count: std::sync::atomic::AtomicU64,
    /// PARITY-003: Set of weight names that have been cached on GPU
    /// Used to avoid repeated dequantization for the same weight
    #[cfg(feature = "cuda")]
    pub(crate) cached_weight_names: std::sync::Mutex<std::collections::HashSet<String>>,
}

// Manual Debug implementation (skip CUDA executor which doesn't impl Debug)
impl std::fmt::Debug for OwnedQuantizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("OwnedQuantizedModel");
        s.field("config", &self.config)
            .field("token_embedding_len", &self.token_embedding.len())
            .field("layers_count", &self.layers.len())
            .field("output_norm_weight_len", &self.output_norm_weight.len())
            .field("has_output_norm_bias", &self.output_norm_bias.is_some())
            .field("lm_head_weight", &self.lm_head_weight)
            .field("has_lm_head_bias", &self.lm_head_bias.is_some());

        #[cfg(feature = "cuda")]
        s.field("cuda_enabled", &self.cuda_executor.is_some())
            .field(
                "cuda_kernel_count",
                &self
                    .cuda_kernel_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "cached_weight_count",
                &self
                    .cached_weight_names
                    .lock()
                    .map(|g| g.len())
                    .unwrap_or(0),
            );

        s.finish()
    }
}

// Manual Clone implementation due to Mutex
impl Clone for OwnedQuantizedModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            layers: self.layers.clone(),
            output_norm_weight: self.output_norm_weight.clone(),
            output_norm_bias: self.output_norm_bias.clone(),
            lm_head_weight: self.lm_head_weight.clone(),
            lm_head_bias: self.lm_head_bias.clone(),
            // CUDA executor is not cloned - new instance must enable CUDA separately
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_transformer_struct() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            hidden_dim: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 256000], // 1000 vocab * 256 hidden
            layers: vec![],
            output_norm_weight: vec![1.0; 256],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 256000],
            lm_head_bias: None,
        };

        assert_eq!(transformer.config.architecture, "llama");
        assert_eq!(transformer.token_embedding.len(), 256000);
        assert!(transformer.layers.is_empty());
    }

    #[test]
    fn test_gguf_transformer_layer_struct() {
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 256],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 256 * 768], // 3 * hidden_dim
            qkv_bias: None,
            attn_output_weight: vec![0.0; 256 * 256],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 256 * 512]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 256 * 512],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 512 * 256],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 256]),
            ffn_norm_bias: None,
        };

        assert_eq!(layer.attn_norm_weight.len(), 256);
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    #[test]
    fn test_owned_quantized_model_clone() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1; 6400],
            layers: vec![],
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 64,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(5),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let cloned = model.clone();
        assert_eq!(cloned.config.architecture, "test");
        assert_eq!(cloned.token_embedding.len(), 6400);

        // CUDA executor is not cloned
        #[cfg(feature = "cuda")]
        {
            assert!(cloned.cuda_executor.is_none());
            assert_eq!(
                cloned
                    .cuda_kernel_count
                    .load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }
    }

    #[test]
    fn test_owned_quantized_model_debug() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "debug_test".to_string(),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.0; 1600],
            layers: vec![],
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: Some(vec![0.0; 32]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![],
                in_dim: 32,
                out_dim: 50,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("token_embedding_len"));
        assert!(debug_str.contains("1600"));
    }
}
