//! CUDA PTX Generation Module
//!
//! Provides NVIDIA CUDA-specific PTX code generation via `trueno-gpu`.
//! This is an optional backend for maximum performance on NVIDIA hardware.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |   CudaKernels API     |  <- Safe public API
//! +-----------------------+
//! |   trueno_gpu::kernels |  <- Hand-optimized PTX kernels
//! +-----------------------+
//! |   trueno_gpu::ptx     |  <- Pure Rust PTX generation
//! +-----------------------+
//! |   CUDA Driver API     |  <- Runtime execution (optional)
//! +-----------------------+
//! ```
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, tensor core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention-style tiled attention
//! - **Quantize**: Q4_K dequantization-fused GEMM
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::cuda::{CudaKernels, KernelType};
//!
//! // Generate PTX for Q4_K quantized GEMM
//! let kernels = CudaKernels::new();
//! let ptx = kernels.generate_ptx(KernelType::QuantizedGemm { m: 1024, n: 1024, k: 4096 });
//!
//! // PTX can be loaded by CUDA driver API
//! println!("{}", ptx);
//! ```

use trueno_gpu::kernels::{
    AttentionKernel, GemmKernel, Kernel, LayerNormKernel, QuantizeKernel, SoftmaxKernel,
};

/// CUDA kernel types supported by realizar
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Naive GEMM (simple, for reference)
    GemmNaive {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
    },
    /// Tiled GEMM with shared memory
    GemmTiled {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
        /// Tile size
        tile_size: u32,
    },
    /// Tensor Core GEMM (fp16)
    GemmTensorCore {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
    },
    /// Numerically stable softmax
    Softmax {
        /// Vector dimension
        dim: u32,
    },
    /// Layer normalization
    LayerNorm {
        /// Hidden dimension
        hidden_size: u32,
        /// Epsilon for numerical stability
        epsilon: f32,
        /// Whether to use affine transform (gamma/beta)
        affine: bool,
    },
    /// FlashAttention-style attention
    Attention {
        /// Sequence length
        seq_len: u32,
        /// Head dimension
        head_dim: u32,
        /// Whether to use causal masking
        causal: bool,
    },
    /// Q4_K quantized GEMM (fused dequantization)
    QuantizedGemm {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension (must be divisible by 32)
        k: u32,
    },
}

/// CUDA kernel generator
///
/// Generates PTX assembly for various GPU kernels using trueno-gpu.
pub struct CudaKernels {
    _private: (),
}

impl CudaKernels {
    /// Create a new CUDA kernel generator
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate PTX source for the specified kernel
    ///
    /// Returns PTX assembly that can be loaded by the CUDA driver API.
    #[must_use]
    pub fn generate_ptx(&self, kernel_type: &KernelType) -> String {
        match kernel_type {
            KernelType::GemmNaive { m, n, k } => GemmKernel::naive(*m, *n, *k).emit_ptx(),
            KernelType::GemmTiled { m, n, k, tile_size } => {
                GemmKernel::tiled(*m, *n, *k, *tile_size).emit_ptx()
            },
            KernelType::GemmTensorCore { m, n, k } => {
                GemmKernel::tensor_core(*m, *n, *k).emit_ptx()
            },
            KernelType::Softmax { dim } => SoftmaxKernel::new(*dim).emit_ptx(),
            KernelType::LayerNorm {
                hidden_size,
                epsilon,
                affine,
            } => {
                let mut kernel = LayerNormKernel::new(*hidden_size);
                if (*epsilon - 1e-5).abs() > f32::EPSILON {
                    kernel = kernel.with_epsilon(*epsilon);
                }
                if !affine {
                    kernel = kernel.without_affine();
                }
                kernel.emit_ptx()
            },
            KernelType::Attention {
                seq_len,
                head_dim,
                causal,
            } => {
                let mut kernel = AttentionKernel::new(*seq_len, *head_dim);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx()
            },
            KernelType::QuantizedGemm { m, n, k } => QuantizeKernel::new(*m, *n, *k).emit_ptx(),
        }
    }

    /// Get kernel name for the specified type
    #[must_use]
    pub fn kernel_name(&self, kernel_type: &KernelType) -> &'static str {
        match kernel_type {
            KernelType::GemmNaive { .. } => "gemm_naive",
            KernelType::GemmTiled { .. } => "gemm_tiled",
            KernelType::GemmTensorCore { .. } => "gemm_tensor_core",
            KernelType::Softmax { .. } => "softmax_warp",
            KernelType::LayerNorm { .. } => "layernorm",
            KernelType::Attention { .. } => "flash_attention",
            KernelType::QuantizedGemm { .. } => "q4k_gemm_fused",
        }
    }

    /// Check if CUDA is likely available on this system
    ///
    /// Note: This is a heuristic check. Actual CUDA availability requires
    /// driver API initialization.
    #[must_use]
    pub fn cuda_likely_available() -> bool {
        // Check for NVIDIA GPU indicators
        std::path::Path::new("/dev/nvidia0").exists()
            || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }
}

impl Default for CudaKernels {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-configured kernel configurations for common LLM inference patterns
pub mod presets {
    use super::KernelType;

    /// Kernel preset for Llama-style attention
    pub fn llama_attention(seq_len: u32, head_dim: u32) -> KernelType {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal: true,
        }
    }

    /// Kernel preset for feed-forward network GEMM
    pub fn ffn_gemm(batch: u32, hidden: u32, intermediate: u32) -> KernelType {
        KernelType::GemmTiled {
            m: batch,
            n: intermediate,
            k: hidden,
            tile_size: 32,
        }
    }

    /// Kernel preset for Q4_K quantized model
    pub fn q4k_inference(batch: u32, hidden: u32, k: u32) -> KernelType {
        KernelType::QuantizedGemm {
            m: batch,
            n: hidden,
            k,
        }
    }

    /// Kernel preset for RMSNorm (LayerNorm variant)
    pub fn rmsnorm(hidden_size: u32) -> KernelType {
        KernelType::LayerNorm {
            hidden_size,
            epsilon: 1e-6,
            affine: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernels_creation() {
        let kernels = CudaKernels::new();
        // Verify the struct was created (ZST is valid)
        let _ = kernels.generate_ptx(&KernelType::Softmax { dim: 128 });
    }

    #[test]
    fn test_gemm_naive_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::GemmNaive {
            m: 128,
            n: 128,
            k: 128,
        });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".visible .entry"));
        assert!(ptx.contains("gemm"));
    }

    #[test]
    fn test_gemm_tiled_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::GemmTiled {
            m: 1024,
            n: 1024,
            k: 1024,
            tile_size: 32,
        });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains("gemm"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_softmax_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 4096 });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains("softmax"));
        assert!(ptx.contains("shfl")); // Warp shuffle
    }

    #[test]
    fn test_layernorm_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
            affine: true,
        });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains("layernorm"));
    }

    #[test]
    fn test_attention_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::Attention {
            seq_len: 2048,
            head_dim: 64,
            causal: true,
        });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains("flash_attention") || ptx.contains("attention"));
    }

    #[test]
    fn test_quantized_gemm_ptx_generation() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
            m: 1,
            n: 4096,
            k: 4096,
        });

        assert!(ptx.contains(".version"));
        assert!(ptx.contains("q4k") || ptx.contains("gemm"));
    }

    #[test]
    fn test_kernel_names() {
        let kernels = CudaKernels::new();

        assert_eq!(
            kernels.kernel_name(&KernelType::GemmNaive { m: 1, n: 1, k: 1 }),
            "gemm_naive"
        );
        assert_eq!(
            kernels.kernel_name(&KernelType::Softmax { dim: 1 }),
            "softmax_warp"
        );
        assert_eq!(
            kernels.kernel_name(&KernelType::QuantizedGemm { m: 1, n: 1, k: 32 }),
            "q4k_gemm_fused"
        );
    }

    #[test]
    fn test_presets_llama_attention() {
        let kernel = presets::llama_attention(2048, 64);
        match kernel {
            KernelType::Attention {
                seq_len,
                head_dim,
                causal,
            } => {
                assert_eq!(seq_len, 2048);
                assert_eq!(head_dim, 64);
                assert!(causal);
            },
            _ => panic!("Expected Attention kernel"),
        }
    }

    #[test]
    fn test_presets_ffn_gemm() {
        let kernel = presets::ffn_gemm(32, 4096, 11008);
        match kernel {
            KernelType::GemmTiled { m, n, k, tile_size } => {
                assert_eq!(m, 32);
                assert_eq!(n, 11008);
                assert_eq!(k, 4096);
                assert_eq!(tile_size, 32);
            },
            _ => panic!("Expected GemmTiled kernel"),
        }
    }

    #[test]
    fn test_presets_q4k_inference() {
        let kernel = presets::q4k_inference(1, 4096, 4096);
        match kernel {
            KernelType::QuantizedGemm { m, n, k } => {
                assert_eq!(m, 1);
                assert_eq!(n, 4096);
                assert_eq!(k, 4096);
            },
            _ => panic!("Expected QuantizedGemm kernel"),
        }
    }

    #[test]
    fn test_presets_rmsnorm() {
        let kernel = presets::rmsnorm(4096);
        match kernel {
            KernelType::LayerNorm {
                hidden_size,
                epsilon,
                affine,
            } => {
                assert_eq!(hidden_size, 4096);
                assert!((epsilon - 1e-6).abs() < 1e-10);
                assert!(!affine);
            },
            _ => panic!("Expected LayerNorm kernel"),
        }
    }

    #[test]
    fn test_default_impl() {
        let kernels = CudaKernels::default();
        let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 256 });
        assert!(!ptx.is_empty());
    }
}
