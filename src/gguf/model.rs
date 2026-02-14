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

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::config::GGUFConfig;
use super::quantized::{OwnedQuantizedLayer, OwnedQuantizedTensor};
use super::types::GGUFModel;
use crate::error::{RealizarError, Result};

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

impl MappedGGUFModel {
    /// Load GGUF model via memory mapping (zero-copy)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to GGUF model file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be opened
    /// - Memory mapping fails
    /// - GGUF parsing fails (invalid format)
    ///
    /// # Performance
    ///
    /// Memory-mapped loading is faster than `std::fs::read` for large models:
    /// - No file content copy to heap memory
    /// - Kernel handles page management
    /// - Model remains accessible even if larger than RAM (via swap)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = MappedGGUFModel::from_path("phi-2-q4_k_m.gguf")?;
    /// println!("Loaded {} tensors", model.model.tensors.len());
    /// ```
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "open_model_file".to_string(),
            reason: format!("Failed to open {}: {}", path.as_ref().display(), e),
        })?;

        // SAFETY: Memory mapping is safe as long as the file isn't modified
        // while mapped. We only read from the mapping, never write.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "mmap_model_file".to_string(),
                reason: format!("Failed to mmap {}: {}", path.as_ref().display(), e),
            })?
        };

        // Parse the memory-mapped data
        let model = GGUFModel::from_bytes(&mmap)?;

        Ok(Self { model, mmap })
    }

    /// Get the raw memory-mapped file data
    ///
    /// This provides direct access to the file contents without copying.
    /// Use this with tensor offsets to read quantized weights directly.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get tensor data slice by offset and size
    ///
    /// Returns a slice pointing directly into the memory-mapped file.
    /// No data is copied.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from start of file
    /// * `size` - Size in bytes
    ///
    /// # Returns
    ///
    /// Slice of tensor data, or None if out of bounds
    #[must_use]
    pub fn tensor_slice(&self, offset: usize, size: usize) -> Option<&[u8]> {
        let end = offset.checked_add(size)?;
        if end <= self.mmap.len() {
            Some(&self.mmap[offset..end])
        } else {
            None
        }
    }

    /// Get the size of the memory-mapped file
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Advise kernel to prefetch model data sequentially
    ///
    /// Per llama.cpp: Use madvise(MADV_SEQUENTIAL) to hint that the model
    /// will be read sequentially during loading. This improves prefetching.
    #[cfg(unix)]
    pub fn advise_sequential(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    /// Advise kernel for random access pattern during inference
    ///
    /// Per llama.cpp: Use madvise(MADV_RANDOM) during inference when
    /// accessing weights non-sequentially.
    #[cfg(unix)]
    pub fn advise_random(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_RANDOM,
            );
        }
    }

    /// Advise kernel to keep model in memory (reduce swap pressure)
    ///
    /// Per llama.cpp: Use madvise(MADV_WILLNEED) to hint that the model
    /// will be needed soon, triggering prefetch.
    #[cfg(unix)]
    pub fn advise_willneed(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Lock model in memory to prevent swapping (requires privileges)
    ///
    /// Per llama.cpp: Use mlock() to ensure model stays in RAM.
    /// Returns true if successful, false if failed (often due to ulimit).
    #[cfg(unix)]
    pub fn lock_memory(&self) -> bool {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe { libc::mlock(self.mmap.as_ptr().cast::<libc::c_void>(), self.mmap.len()) == 0 }
    }
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
            bos_token_id: None,
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
            bos_token_id: None,
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
            bos_token_id: None,
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

    #[test]
    fn test_gguf_transformer_with_biases() {
        let config = GGUFConfig {
            architecture: "phi".to_string(),
            hidden_dim: 128,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 500,
            intermediate_dim: 256,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 64000],
            layers: vec![],
            output_norm_weight: vec![1.0; 128],
            output_norm_bias: Some(vec![0.0; 128]),
            lm_head_weight: vec![0.0; 64000],
            lm_head_bias: Some(vec![0.0; 500]),
        };

        assert!(transformer.output_norm_bias.is_some());
        assert!(transformer.lm_head_bias.is_some());
        assert_eq!(transformer.lm_head_bias.as_ref().unwrap().len(), 500);
    }

    #[test]
    fn test_gguf_transformer_with_layers() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        let layer1 = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: None,
        };

        let layer2 = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: None,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 6400],
            layers: vec![layer1, layer2],
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 6400],
            lm_head_bias: None,
        };

        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.layers[0].qkv_weight.len(), 64 * 192);
    }

    #[test]
    fn test_gguf_transformer_layer_without_gate() {
        // Models like phi-2 don't use SwiGLU, so no gate
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 128],
            attn_norm_bias: Some(vec![0.0; 128]),
            qkv_weight: vec![0.0; 128 * 384],
            qkv_bias: Some(vec![0.0; 384]),
            attn_output_weight: vec![0.0; 128 * 128],
            attn_output_bias: Some(vec![0.0; 128]),
            ffn_gate_weight: None, // No gate for phi-2 style
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 128 * 512],
            ffn_up_bias: Some(vec![0.0; 512]),
            ffn_down_weight: vec![0.0; 512 * 128],
            ffn_down_bias: Some(vec![0.0; 128]),
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };

        assert!(layer.ffn_gate_weight.is_none());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
    }

    #[test]
    fn test_gguf_transformer_layer_all_biases() {
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: Some(vec![0.0; 64]),
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: Some(vec![0.0; 192]),
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: Some(vec![0.0; 64]),
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: Some(vec![0.0; 128]),
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: Some(vec![0.0; 128]),
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: Some(vec![0.0; 64]),
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: Some(vec![0.0; 64]),
        };

        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
        assert!(layer.ffn_gate_bias.is_some());
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());
        assert!(layer.ffn_norm_bias.is_some());
    }

    #[test]
    fn test_owned_quantized_model_with_bias() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "phi".to_string(),
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
            bos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1; 6400],
            layers: vec![],
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: Some(vec![0.0; 64]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 64,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: Some(vec![0.0; 100]),
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        assert!(model.output_norm_bias.is_some());
        assert!(model.lm_head_bias.is_some());
        assert_eq!(model.lm_head_bias.as_ref().unwrap().len(), 100);
    }

    #[test]
    fn test_owned_quantized_model_clone_preserves_data() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q6_K;

        let config = GGUFConfig {
            architecture: "test_clone".to_string(),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
            rope_type: 0,
            bos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.5, 0.6, 0.7, 0.8],
            layers: vec![],
            output_norm_weight: vec![1.0, 1.0, 1.0],
            output_norm_bias: Some(vec![0.1, 0.2, 0.3]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![1, 2, 3, 4, 5],
                in_dim: 32,
                out_dim: 50,
                qtype: GGUF_TYPE_Q6_K,
            },
            lm_head_bias: Some(vec![0.0; 50]),
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(10),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let cloned = model.clone();

        // Verify all data is cloned
        assert_eq!(cloned.token_embedding, model.token_embedding);
        assert_eq!(cloned.output_norm_weight, model.output_norm_weight);
        assert_eq!(cloned.output_norm_bias, model.output_norm_bias);
        assert_eq!(cloned.lm_head_weight.data, model.lm_head_weight.data);
        assert_eq!(cloned.lm_head_weight.qtype, GGUF_TYPE_Q6_K);
        assert_eq!(cloned.config.architecture, "test_clone");
    }

    #[test]
    fn test_owned_quantized_model_debug_shows_all_fields() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q8_0;

        let config = GGUFConfig {
            architecture: "llama_debug".to_string(),
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 256,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.0; 128000],
            layers: vec![],
            output_norm_weight: vec![1.0; 128],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 1000],
                in_dim: 128,
                out_dim: 1000,
                qtype: GGUF_TYPE_Q8_0,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(42),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("llama_debug"));
        assert!(debug_str.contains("token_embedding_len"));
        assert!(debug_str.contains("128000"));
        assert!(debug_str.contains("layers_count"));
        assert!(debug_str.contains("output_norm_weight_len"));
        assert!(debug_str.contains("has_output_norm_bias"));
        assert!(debug_str.contains("lm_head_weight"));
        assert!(debug_str.contains("has_lm_head_bias"));
        #[cfg(feature = "cuda")]
        assert!(debug_str.contains("cuda_enabled"));
    }

    #[test]
    fn test_gguf_config_fields() {
        let config = GGUFConfig {
            architecture: "mistral".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA style
            vocab_size: 32000,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 1000000.0,
            eps: 1e-5,
            rope_type: 1,
            bos_token_id: None,
        };

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 14336);
        assert_eq!(config.context_length, 8192);
        assert!((config.rope_theta - 1000000.0).abs() < 0.1);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert_eq!(config.rope_type, 1);
    }

    #[test]
    fn test_mapped_gguf_model_file_not_found() {
        let result = MappedGGUFModel::from_path("/nonexistent/path/model.gguf");
        assert!(result.is_err());
    }

    // =========================================================================
    // GH-40 FALSIFICATION: OwnedQuantizedModel API contract stability
    // =========================================================================

    /// GH-40: OwnedQuantizedModel must expose all public fields required by
    /// the inference pipeline. If a field is renamed, removed, or made private,
    /// this test fails — catching API breakage at compile time.
    #[test]
    fn test_falsify_gh40_api_contract_all_fields_accessible() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "gh40_test".to_string(),
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
            bos_token_id: None,
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
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        // GH-40 contract: every field below must be pub and have the expected type.
        // If ANY of these lines fails to compile, the API contract is broken.
        let _arch: &str = &model.config.architecture;
        let _hidden: usize = model.config.hidden_dim;
        let _embed: &[f32] = &model.token_embedding;
        let _layers: &[super::super::quantized::OwnedQuantizedLayer] = &model.layers;
        let _norm: &[f32] = &model.output_norm_weight;
        let _norm_bias: &Option<Vec<f32>> = &model.output_norm_bias;
        let _lm_data: &[u8] = &model.lm_head_weight.data;
        let _lm_in: usize = model.lm_head_weight.in_dim;
        let _lm_out: usize = model.lm_head_weight.out_dim;
        let _lm_qt: u32 = model.lm_head_weight.qtype;
        let _lm_bias: &Option<Vec<f32>> = &model.lm_head_bias;

        // GH-40 contract: config fields required by inference
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.config.num_heads, 2);
        assert_eq!(model.config.num_kv_heads, 2);
        assert_eq!(model.config.vocab_size, 100);
        assert_eq!(model.config.intermediate_dim, 128);
        assert!(model.config.rope_theta > 0.0);
        assert!(model.config.eps > 0.0);
    }

    /// GH-40: GGUFConfig must have bos_token_id field (added in the fix).
    /// Without this field, tokenizer initialization fails silently.
    #[test]
    fn test_falsify_gh40_config_has_bos_token_id() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: Some(1),
        };
        assert_eq!(
            config.bos_token_id,
            Some(1),
            "GH-40: bos_token_id must be accessible"
        );
    }
}
