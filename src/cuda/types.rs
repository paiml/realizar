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
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightQuantType {
    /// Q4_K quantization (type 12) - 144 bytes per 256 elements
    #[default]
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
            }
            // Block formats (32 elements per block)
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q8_0 => {
                let n_blocks = n_rows * ((n_cols + 31) / 32);
                size_bytes == n_blocks * self.bytes_per_block()
            }
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
