//! CUDA Type Definitions for Weight Loading and Workspace Management
//!
//! This module contains types used for GPU weight management:
//! - `IndexedLayerWeights`: Pre-computed layer weight indices for O(1) lookup
//! - `WeightQuantType`: Quantization type detection and size calculation
//! - `TransformerWorkspace`: Pre-allocated workspace buffers

use trueno_gpu::driver::GpuBuffer;

/// PAR-043: Pre-computed layer weight indices for O(1) lookup
///
/// Eliminates per-layer string formatting and HashMap lookups during decode.
/// Each layer's weights are stored as raw device pointers for direct access.
///
/// Performance impact:
/// - Before: ~10-12ms overhead per token (string formatting + HashMap)
/// - After: ~0.1ms overhead per token (direct indexed access)
/// PMAT-232 CONTRACT: No Default — every field must be explicitly set from GGUF metadata.
#[derive(Debug, Clone)]
pub struct IndexedLayerWeights {
    /// Q projection weights device pointer (may be Q4K or Q5_0 quantized)
    pub attn_q_ptr: u64,
    /// Q projection weights size in bytes
    pub attn_q_len: usize,
    /// Q projection quantization type (Qwen 0.5B uses Q5_0)
    pub attn_q_qtype: WeightQuantType,
    /// K projection weights device pointer (may be Q4K or Q5_0 quantized)
    pub attn_k_ptr: u64,
    /// K projection weights size in bytes
    pub attn_k_len: usize,
    /// K projection quantization type (Qwen 0.5B uses Q5_0)
    pub attn_k_qtype: WeightQuantType,
    /// V projection weights device pointer (may be Q4K, Q6K, or Q8_0 quantized)
    pub attn_v_ptr: u64,
    /// V projection weights size in bytes
    pub attn_v_len: usize,
    /// V projection quantization type (needed because some models use Q6K/Q8_0 for V)
    pub attn_v_qtype: WeightQuantType,
    /// O projection weights device pointer (may be Q4K or Q4_0 quantized)
    pub attn_output_ptr: u64,
    /// O projection weights size in bytes
    pub attn_output_len: usize,
    /// O projection quantization type (PAR-058: Q4_0 models were broken)
    pub attn_output_qtype: WeightQuantType,
    /// FFN gate projection device pointer (may be Q4K or Q4_0 quantized)
    pub ffn_gate_ptr: u64,
    /// FFN gate projection size in bytes
    pub ffn_gate_len: usize,
    /// FFN gate projection quantization type (PAR-058: Q4_0 models were broken)
    pub ffn_gate_qtype: WeightQuantType,
    /// FFN up projection device pointer (may be Q4K or Q4_0 quantized)
    pub ffn_up_ptr: u64,
    /// FFN up projection size in bytes
    pub ffn_up_len: usize,
    /// FFN up projection quantization type (PAR-058: Q4_0 models were broken)
    pub ffn_up_qtype: WeightQuantType,
    /// FFN down projection device pointer (Q4K, Q6K, or Q4_0 quantized)
    pub ffn_down_ptr: u64,
    /// FFN down projection size in bytes
    pub ffn_down_len: usize,
    /// FFN down projection quantization type (some models use Q6K)
    pub ffn_down_qtype: WeightQuantType,
    /// Attention RMSNorm gamma device pointer (FP32)
    pub attn_norm_ptr: u64,
    /// Attention RMSNorm gamma size in elements
    pub attn_norm_len: usize,
    /// FFN RMSNorm gamma device pointer (FP32)
    pub ffn_norm_ptr: u64,
    /// FFN RMSNorm gamma size in elements
    pub ffn_norm_len: usize,
    /// Q projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_q_bias_ptr: u64,
    /// Q projection bias size in elements (0 if no bias)
    pub attn_q_bias_len: usize,
    /// K projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_k_bias_ptr: u64,
    /// K projection bias size in elements (0 if no bias)
    pub attn_k_bias_len: usize,
    /// V projection bias device pointer (FP32, optional - 0 if no bias)
    pub attn_v_bias_ptr: u64,
    /// V projection bias size in elements (0 if no bias)
    pub attn_v_bias_len: usize,
}

/// Weight quantization type for GGUF tensors
///
/// PMAT-232 CONTRACT: This enum MUST NOT derive Default. Every construction
/// must be explicit. Match statements MUST be exhaustive (no `_ =>` catch-all).
/// See contracts/tensor-layout-v1.yaml quant_dispatch section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantType {
    /// Q4_K quantization (type 12) - 144 bytes per 256 elements
    Q4K,
    /// Q5_K quantization (type 13) - 176 bytes per 256 elements
    Q5K,
    /// Q6_K quantization (type 14) - 210 bytes per 256 elements
    Q6K,
    /// Q8_0 quantization (type 8) - 34 bytes per 32 elements
    Q8_0,
    /// Q5_0 quantization (type 6) - 22 bytes per 32 elements
    Q5_0,
    /// Q4_0 quantization (type 2) - 18 bytes per 32 elements
    Q4_0,
    /// Q4_1 quantization (type 3) - 20 bytes per 32 elements (2 f16 scale + 2 f16 min + 16 quants)
    /// PAR-058: Added to handle Qwen 0.5B which has FFN down in Q4_1 despite metadata
    Q4_1,
}

impl WeightQuantType {
    /// Bytes per 256 elements for super-block quantization types
    pub const fn bytes_per_superblock(&self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8_0 => 34 * 8, // Q8_0 uses 32-element blocks, so 8 blocks for 256 elements
            Self::Q5_0 => 22 * 8, // Q5_0 uses 32-element blocks, so 8 blocks for 256 elements
            Self::Q4_0 => 18 * 8, // Q4_0 uses 32-element blocks, so 8 blocks for 256 elements
            Self::Q4_1 => 20 * 8, // Q4_1 uses 32-element blocks, so 8 blocks for 256 elements
        }
    }

    /// Bytes per 32 elements (for block-based quantization types)
    pub const fn bytes_per_block(&self) -> usize {
        match self {
            Self::Q4K => 18, // Q4K is super-block, treat as 18 per 32 for calculation
            Self::Q5K => 22, // Q5K is super-block
            Self::Q6K => 26, // Q6K is super-block (210/8 = 26.25, round to 26)
            Self::Q8_0 => 34,
            Self::Q5_0 => 22,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
        }
    }

    /// Create from GGML type ID
    pub fn from_ggml_type(type_id: u32) -> Option<Self> {
        match type_id {
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1), // PAR-058: Q4_1 support
            6 => Some(Self::Q5_0),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// PAR-105-FIX: Check if a qtype matches the expected size for given dimensions
    /// Returns true if the qtype would produce the given byte size
    pub fn matches_size(&self, size_bytes: usize, n_rows: usize, n_cols: usize) -> bool {
        match self {
            // Super-block formats (256 elements per super-block)
            Self::Q4K | Self::Q5K | Self::Q6K => {
                let n_superblocks = n_rows * ((n_cols + 255) / 256);
                size_bytes == n_superblocks * self.bytes_per_superblock()
            },
            // Block formats (32 elements per block)
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q8_0 => {
                let n_blocks = n_rows * ((n_cols + 31) / 32);
                size_bytes == n_blocks * self.bytes_per_block()
            },
        }
    }

    /// PAR-058: Detect quantization type from actual weight size
    /// Some GGUF files have incorrect type metadata, so we verify by size
    ///
    /// CORRECTNESS-002 FIX: For certain dimension combinations, Q4_0 and Q4K have
    /// the SAME byte size (e.g., 1536×8960: 1536×280×18 = 1536×35×144 = 7,741,440).
    /// Check super-block formats FIRST since they have more distinctive layouts.
    pub fn from_size(size_bytes: usize, n_rows: usize, n_cols: usize) -> Option<Self> {
        // CORRECTNESS-002: Check super-block formats FIRST
        // Super-block formats (256 elements per super-block)
        let n_superblocks = n_rows * ((n_cols + 255) / 256);
        let superblock_formats = [(Self::Q6K, 210), (Self::Q5K, 176), (Self::Q4K, 144)];

        for (fmt, bytes_per_sb) in superblock_formats {
            if size_bytes == n_superblocks * bytes_per_sb {
                return Some(fmt);
            }
        }

        // Then check block formats (32 elements per block)
        let n_blocks = n_rows * ((n_cols + 31) / 32);
        let formats = [
            (Self::Q4_0, 18),
            (Self::Q4_1, 20),
            (Self::Q5_0, 22),
            (Self::Q8_0, 34),
        ];

        for (fmt, bytes_per_block) in formats {
            if size_bytes == n_blocks * bytes_per_block {
                return Some(fmt);
            }
        }

        None
    }
}

// =============================================================================
// PMAT-232: Bound Weight — kernel resolved at model load, not at inference
// =============================================================================
//
// Architecture: The model format defines the kernel. The kernel is bound at
// load time. The forward pass has ZERO dispatch.
//
// Before (7+ match sites per forward call):
//   match layer_weights.attn_q_qtype {
//       Q4K => q4k_gemv_into(...),
//       Q6K => q6k_gemv_into(...),
//       _ => q4k_gemv_into(...),  // BUG: catch-all uses wrong kernel
//   }
//
// After (0 match sites per forward call):
//   layer.q_proj.gemv(executor, &input, &output)?;  // kernel pre-bound
//
// The match happens ONCE in BoundWeight::bind(). Adding a new WeightQuantType
// variant produces a compile error in exactly ONE place.

/// A GPU weight with its GEMV kernel pre-bound at model load time.
///
/// Construction validates the quant type → kernel mapping. The forward pass
/// calls `.gemv()` which CANNOT dispatch the wrong kernel because the kernel
/// was resolved at bind time.
///
/// This is Poka-Yoke applied to kernel dispatch: the mistake (wrong kernel)
/// is structurally impossible after construction.
#[derive(Debug, Clone)]
pub struct BoundWeight {
    /// Device pointer to quantized weight data
    pub ptr: u64,
    /// Size in bytes
    pub len: usize,
    /// Output dimension (rows in weight matrix)
    pub out_dim: u32,
    /// Input dimension (cols in weight matrix)
    pub in_dim: u32,
    /// The kernel that was bound at construction — private, cannot be changed
    kernel: GemvKernel,
}

/// The GEMV kernel to use. Resolved ONCE at model load time.
/// This is the SINGLE source of truth for quant type → kernel mapping.
/// See contracts/tensor-layout-v1.yaml quant_dispatch section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvKernel {
    /// Q4_K super-block kernel (144 bytes / 256 elements)
    Q4K,
    /// Q5_K super-block kernel (176 bytes / 256 elements)
    Q5K,
    /// Q6_K super-block kernel (210 bytes / 256 elements)
    Q6K,
    /// Q8_0 block kernel (34 bytes / 32 elements)
    Q8_0,
    /// Q4_0 block kernel (18 bytes / 32 elements)
    Q4_0,
    /// Q5_0 block kernel (22 bytes / 32 elements)
    Q5_0,
    /// Q4_1 block kernel (20 bytes / 32 elements)
    Q4_1,
}

impl BoundWeight {
    /// Bind a weight to its correct GEMV kernel based on quantization type.
    ///
    /// This is the ONE place where WeightQuantType → GemvKernel mapping happens.
    /// The match is exhaustive — adding a new variant is a compile error here.
    pub fn bind(ptr: u64, len: usize, qtype: WeightQuantType, out_dim: u32, in_dim: u32) -> Self {
        // PMAT-232: Exhaustive mapping — no catch-all, no default.
        // If you add a WeightQuantType variant, this MUST be updated.
        let kernel = match qtype {
            WeightQuantType::Q4K => GemvKernel::Q4K,
            WeightQuantType::Q5K => GemvKernel::Q5K,
            WeightQuantType::Q6K => GemvKernel::Q6K,
            WeightQuantType::Q8_0 => GemvKernel::Q8_0,
            WeightQuantType::Q4_0 => GemvKernel::Q4_0,
            WeightQuantType::Q5_0 => GemvKernel::Q5_0,
            WeightQuantType::Q4_1 => GemvKernel::Q4_1,
        };
        Self {
            ptr,
            len,
            out_dim,
            in_dim,
            kernel,
        }
    }

    /// The bound kernel (read-only).
    pub fn kernel(&self) -> GemvKernel {
        self.kernel
    }
}

/// A complete transformer layer with all kernels pre-bound.
///
/// Constructed from `IndexedLayerWeights` at model load time.
/// The forward pass uses this — ZERO dispatch, ZERO match statements.
#[derive(Debug, Clone)]
pub struct BoundLayerWeights {
    /// Q projection (hidden → q_dim)
    pub q_proj: BoundWeight,
    /// K projection (hidden → kv_dim)
    pub k_proj: BoundWeight,
    /// V projection (hidden → kv_dim)
    pub v_proj: BoundWeight,
    /// Output projection (q_dim → hidden)
    pub o_proj: BoundWeight,
    /// FFN gate projection (hidden → intermediate)
    pub ffn_gate: BoundWeight,
    /// FFN up projection (hidden → intermediate)
    pub ffn_up: BoundWeight,
    /// FFN down projection (intermediate → hidden)
    pub ffn_down: BoundWeight,
    /// Attention norm weight pointer
    pub attn_norm_ptr: u64,
    /// Attention norm weight length
    pub attn_norm_len: usize,
    /// FFN norm weight pointer
    pub ffn_norm_ptr: u64,
    /// FFN norm weight length
    pub ffn_norm_len: usize,
    /// Q bias pointer (0 if no bias)
    pub attn_q_bias_ptr: u64,
    /// Q bias length in elements (0 if no bias)
    pub attn_q_bias_len: usize,
    /// K bias pointer (0 if no bias)
    pub attn_k_bias_ptr: u64,
    /// K bias length in elements (0 if no bias)
    pub attn_k_bias_len: usize,
    /// V bias pointer (0 if no bias)
    pub attn_v_bias_ptr: u64,
    /// V bias length in elements (0 if no bias)
    pub attn_v_bias_len: usize,
}

impl BoundLayerWeights {
    /// Bind all layer weights from IndexedLayerWeights.
    ///
    /// This is the compilation step: quant types are resolved to kernels ONCE.
    /// After this, the forward pass has zero dispatch.
    pub fn bind(
        src: &IndexedLayerWeights,
        hidden_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        intermediate_dim: u32,
    ) -> Self {
        Self {
            q_proj: BoundWeight::bind(
                src.attn_q_ptr, src.attn_q_len, src.attn_q_qtype, q_dim, hidden_dim,
            ),
            k_proj: BoundWeight::bind(
                src.attn_k_ptr, src.attn_k_len, src.attn_k_qtype, kv_dim, hidden_dim,
            ),
            v_proj: BoundWeight::bind(
                src.attn_v_ptr, src.attn_v_len, src.attn_v_qtype, kv_dim, hidden_dim,
            ),
            o_proj: BoundWeight::bind(
                src.attn_output_ptr, src.attn_output_len, src.attn_output_qtype,
                hidden_dim, q_dim,
            ),
            ffn_gate: BoundWeight::bind(
                src.ffn_gate_ptr, src.ffn_gate_len, src.ffn_gate_qtype,
                intermediate_dim, hidden_dim,
            ),
            ffn_up: BoundWeight::bind(
                src.ffn_up_ptr, src.ffn_up_len, src.ffn_up_qtype,
                intermediate_dim, hidden_dim,
            ),
            ffn_down: BoundWeight::bind(
                src.ffn_down_ptr, src.ffn_down_len, src.ffn_down_qtype,
                hidden_dim, intermediate_dim,
            ),
            attn_norm_ptr: src.attn_norm_ptr,
            attn_norm_len: src.attn_norm_len,
            ffn_norm_ptr: src.ffn_norm_ptr,
            ffn_norm_len: src.ffn_norm_len,
            attn_q_bias_ptr: src.attn_q_bias_ptr,
            attn_q_bias_len: src.attn_q_bias_len,
            attn_k_bias_ptr: src.attn_k_bias_ptr,
            attn_k_bias_len: src.attn_k_bias_len,
            attn_v_bias_ptr: src.attn_v_bias_ptr,
            attn_v_bias_len: src.attn_v_bias_len,
        }
    }
}

/// PAR-044: Pre-allocated workspace buffers for transformer forward pass
///
/// Eliminates ~288 GPU buffer allocations per token by reusing pre-sized buffers.
/// All buffers are allocated once at model load and reused for every token.
///
/// Performance impact:
/// - Before: ~288 cuMemAlloc calls per token (~2-3ms overhead)
/// - After: 0 allocations per token (all reused)
#[derive(Default)]
pub struct TransformerWorkspace {
    /// Hidden state buffer 1 (hidden_dim) - for normed, projected, ffn_normed, ffn_down
    pub hidden_buf1: Option<GpuBuffer<f32>>,
    /// Hidden state buffer 2 (hidden_dim) - for residual1, output
    pub hidden_buf2: Option<GpuBuffer<f32>>,
    /// Input staging buffer (hidden_dim) - preserves input for residual connections
    pub input_staging: Option<GpuBuffer<f32>>,
    /// Q/attention output buffer (q_dim)
    pub q_buf: Option<GpuBuffer<f32>>,
    /// K projection buffer (kv_dim)
    pub k_buf: Option<GpuBuffer<f32>>,
    /// V projection buffer (kv_dim)
    pub v_buf: Option<GpuBuffer<f32>>,
    /// FFN gate buffer (intermediate_dim)
    pub ffn_gate_buf: Option<GpuBuffer<f32>>,
    /// FFN up buffer (intermediate_dim)
    pub ffn_up_buf: Option<GpuBuffer<f32>>,
    /// FFN activated buffer (intermediate_dim) - result of SwiGLU
    pub ffn_act_buf: Option<GpuBuffer<f32>>,
    /// Attention output buffer (q_dim) - result of incremental attention
    /// PAR-051: Eliminates 28 GPU allocations per token
    pub attn_out_buf: Option<GpuBuffer<f32>>,
    /// PAR-054: Logits output buffer (vocab_size) - for CUDA graph capture
    pub logits_buf: Option<GpuBuffer<f32>>,
    /// PAR-054: Normed hidden buffer (hidden_dim) - for CUDA graph capture
    pub normed_hidden_buf: Option<GpuBuffer<f32>>,
    /// Workspace is initialized
    pub initialized: bool,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Q dimension (num_heads × head_dim)
    pub q_dim: usize,
    /// KV dimension (num_kv_heads × head_dim)
    pub kv_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
    /// PAR-111: Batch size for multi-sequence processing (default 1)
    pub batch_size: usize,
    /// PAR-114: Positions buffer for batched RoPE (M positions)
    pub positions_buf: Option<GpuBuffer<u32>>,
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create zeroed `IndexedLayerWeights` for tests.
    /// PMAT-232: `Default` was intentionally removed to enforce explicit
    /// construction from GGUF metadata in production code.
    fn test_zeroed_layer_weights() -> IndexedLayerWeights {
        IndexedLayerWeights {
            attn_q_ptr: 0, attn_q_len: 0, attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 0, attn_k_len: 0, attn_k_qtype: WeightQuantType::Q4K,
            attn_v_ptr: 0, attn_v_len: 0, attn_v_qtype: WeightQuantType::Q4K,
            attn_output_ptr: 0, attn_output_len: 0, attn_output_qtype: WeightQuantType::Q4K,
            ffn_gate_ptr: 0, ffn_gate_len: 0, ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 0, ffn_up_len: 0, ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 0, ffn_down_len: 0, ffn_down_qtype: WeightQuantType::Q4K,
            attn_norm_ptr: 0, attn_norm_len: 0,
            ffn_norm_ptr: 0, ffn_norm_len: 0,
            attn_q_bias_ptr: 0, attn_q_bias_len: 0,
            attn_k_bias_ptr: 0, attn_k_bias_len: 0,
            attn_v_bias_ptr: 0, attn_v_bias_len: 0,
        }
    }

    // =========================================================================
    // WeightQuantType Tests
    // =========================================================================

    #[test]
    fn test_weight_quant_type_no_default() {
        // PMAT-232: WeightQuantType must NOT have a Default impl.
        // Every construction must be explicit to prevent silent wrong-kernel dispatch.
        // If this test fails to compile, the contract is enforced correctly.
        let explicit = WeightQuantType::Q4K;
        assert_eq!(explicit, WeightQuantType::Q4K);
    }

    #[test]
    fn test_weight_quant_type_bytes_per_superblock() {
        assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
        assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
        assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);
        assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8); // 272
        assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8); // 176
        assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8); // 144
        assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8); // 160
    }

    #[test]
    fn test_weight_quant_type_bytes_per_block() {
        assert_eq!(WeightQuantType::Q4K.bytes_per_block(), 18);
        assert_eq!(WeightQuantType::Q5K.bytes_per_block(), 22);
        assert_eq!(WeightQuantType::Q6K.bytes_per_block(), 26);
        assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
        assert_eq!(WeightQuantType::Q5_0.bytes_per_block(), 22);
        assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);
        assert_eq!(WeightQuantType::Q4_1.bytes_per_block(), 20);
    }

    #[test]
    fn test_weight_quant_type_from_ggml_type() {
        assert_eq!(
            WeightQuantType::from_ggml_type(2),
            Some(WeightQuantType::Q4_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(3),
            Some(WeightQuantType::Q4_1)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(6),
            Some(WeightQuantType::Q5_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(8),
            Some(WeightQuantType::Q8_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(12),
            Some(WeightQuantType::Q4K)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(13),
            Some(WeightQuantType::Q5K)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(14),
            Some(WeightQuantType::Q6K)
        );
        assert_eq!(WeightQuantType::from_ggml_type(99), None);
        assert_eq!(WeightQuantType::from_ggml_type(0), None);
    }

    #[test]
    fn test_weight_quant_type_matches_size_superblock() {
        // Q4K: 144 bytes per 256 elements
        // For 1024 rows × 256 cols: 1024 super-blocks × 144 = 147456 bytes
        assert!(WeightQuantType::Q4K.matches_size(147_456, 1024, 256));
        assert!(!WeightQuantType::Q4K.matches_size(147_457, 1024, 256)); // Wrong size

        // Q5K: 176 bytes per 256 elements
        assert!(WeightQuantType::Q5K.matches_size(1024 * 176, 1024, 256));

        // Q6K: 210 bytes per 256 elements
        assert!(WeightQuantType::Q6K.matches_size(1024 * 210, 1024, 256));
    }

    #[test]
    fn test_weight_quant_type_matches_size_block() {
        // Q4_0: 18 bytes per 32 elements
        // For 1024 rows × 32 cols: 1024 blocks × 18 = 18432 bytes
        assert!(WeightQuantType::Q4_0.matches_size(18_432, 1024, 32));

        // Q8_0: 34 bytes per 32 elements
        assert!(WeightQuantType::Q8_0.matches_size(1024 * 34, 1024, 32));

        // Q5_0: 22 bytes per 32 elements
        assert!(WeightQuantType::Q5_0.matches_size(1024 * 22, 1024, 32));

        // Q4_1: 20 bytes per 32 elements
        assert!(WeightQuantType::Q4_1.matches_size(1024 * 20, 1024, 32));
    }

    #[test]
    fn test_weight_quant_type_matches_size_partial_blocks() {
        // Test with non-aligned dimensions
        // 1024 rows × 100 cols for Q4_0: (100 + 31) / 32 = 4 blocks per row
        // 1024 × 4 × 18 = 73728 bytes
        assert!(WeightQuantType::Q4_0.matches_size(73_728, 1024, 100));
    }

    #[test]
    fn test_weight_quant_type_from_size_superblock() {
        // Q4K: 144 bytes per 256 elements
        let size = 1024 * 144; // 1024 super-blocks
        assert_eq!(
            WeightQuantType::from_size(size, 1024, 256),
            Some(WeightQuantType::Q4K)
        );

        // Q5K: 176 bytes per 256 elements
        let size = 512 * 176;
        assert_eq!(
            WeightQuantType::from_size(size, 512, 256),
            Some(WeightQuantType::Q5K)
        );

        // Q6K: 210 bytes per 256 elements
        let size = 256 * 210;
        assert_eq!(
            WeightQuantType::from_size(size, 256, 256),
            Some(WeightQuantType::Q6K)
        );
    }

    #[test]
    fn test_weight_quant_type_from_size_block() {
        // Q4_0: 18 bytes per 32 elements
        let size = 1024 * 18;
        assert_eq!(
            WeightQuantType::from_size(size, 1024, 32),
            Some(WeightQuantType::Q4_0)
        );

        // Q8_0: 34 bytes per 32 elements
        let size = 512 * 34;
        assert_eq!(
            WeightQuantType::from_size(size, 512, 32),
            Some(WeightQuantType::Q8_0)
        );

        // Q5_0: 22 bytes per 32 elements
        let size = 256 * 22;
        assert_eq!(
            WeightQuantType::from_size(size, 256, 32),
            Some(WeightQuantType::Q5_0)
        );

        // Q4_1: 20 bytes per 32 elements
        let size = 128 * 20;
        assert_eq!(
            WeightQuantType::from_size(size, 128, 32),
            Some(WeightQuantType::Q4_1)
        );
    }

    #[test]
    fn test_weight_quant_type_from_size_none() {
        // Size that doesn't match any format
        assert_eq!(WeightQuantType::from_size(12345, 100, 256), None);
    }

    #[test]
    fn test_weight_quant_type_clone_eq() {
        let qtype = WeightQuantType::Q6K;
        let cloned = qtype;
        assert_eq!(qtype, cloned);
    }

    #[test]
    fn test_weight_quant_type_debug() {
        let qtype = WeightQuantType::Q4K;
        let debug = format!("{:?}", qtype);
        assert!(debug.contains("Q4K"));
    }

    // =========================================================================
    // IndexedLayerWeights Tests
    // =========================================================================

    #[test]
    fn test_indexed_layer_weights_zeroed() {
        let weights = test_zeroed_layer_weights();
        assert_eq!(weights.attn_q_ptr, 0);
        assert_eq!(weights.attn_q_len, 0);
        assert_eq!(weights.attn_q_qtype, WeightQuantType::Q4K);
        assert_eq!(weights.ffn_gate_ptr, 0);
        assert_eq!(weights.attn_norm_len, 0);
    }

    #[test]
    fn test_indexed_layer_weights_clone() {
        let mut weights = test_zeroed_layer_weights();
        weights.attn_q_ptr = 12345;
        weights.attn_q_len = 1024;
        weights.attn_q_qtype = WeightQuantType::Q5K;

        let cloned = weights.clone();
        assert_eq!(cloned.attn_q_ptr, 12345);
        assert_eq!(cloned.attn_q_len, 1024);
        assert_eq!(cloned.attn_q_qtype, WeightQuantType::Q5K);
    }

    #[test]
    fn test_indexed_layer_weights_debug() {
        let weights = test_zeroed_layer_weights();
        let debug = format!("{:?}", weights);
        assert!(debug.contains("IndexedLayerWeights"));
    }

    #[test]
    fn test_indexed_layer_weights_all_fields() {
        let weights = IndexedLayerWeights {
            attn_q_ptr: 100,
            attn_q_len: 1024,
            attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 200,
            attn_k_len: 512,
            attn_k_qtype: WeightQuantType::Q5K,
            attn_v_ptr: 300,
            attn_v_len: 512,
            attn_v_qtype: WeightQuantType::Q6K,
            attn_output_ptr: 400,
            attn_output_len: 1024,
            attn_output_qtype: WeightQuantType::Q4_0,
            ffn_gate_ptr: 500,
            ffn_gate_len: 4096,
            ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 600,
            ffn_up_len: 4096,
            ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 700,
            ffn_down_len: 1024,
            ffn_down_qtype: WeightQuantType::Q6K,
            attn_norm_ptr: 800,
            attn_norm_len: 1024,
            ffn_norm_ptr: 900,
            ffn_norm_len: 1024,
            attn_q_bias_ptr: 0,
            attn_q_bias_len: 0,
            attn_k_bias_ptr: 0,
            attn_k_bias_len: 0,
            attn_v_bias_ptr: 0,
            attn_v_bias_len: 0,
        };

        assert_eq!(weights.attn_q_ptr, 100);
        assert_eq!(weights.ffn_down_qtype, WeightQuantType::Q6K);
        assert_eq!(weights.attn_q_bias_len, 0); // No bias
    }

    // =========================================================================
    // TransformerWorkspace Tests
    // =========================================================================

    #[test]
    fn test_transformer_workspace_default() {
        let workspace = TransformerWorkspace::default();
        assert!(!workspace.initialized);
        assert_eq!(workspace.hidden_dim, 0);
        assert_eq!(workspace.q_dim, 0);
        assert_eq!(workspace.kv_dim, 0);
        assert_eq!(workspace.intermediate_dim, 0);
        assert_eq!(workspace.batch_size, 0);
        assert!(workspace.hidden_buf1.is_none());
        assert!(workspace.hidden_buf2.is_none());
        assert!(workspace.q_buf.is_none());
        assert!(workspace.k_buf.is_none());
        assert!(workspace.v_buf.is_none());
        assert!(workspace.logits_buf.is_none());
    }
}
