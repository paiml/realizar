//! APR Transformer Format for WASM-compatible LLM inference
//!
//! This module provides a WASM-compatible transformer implementation that stores
//! all weights as F32, enabling fair comparison between APR and GGUF formats.
//!
//! ## Design Goals
//!
//! 1. **WASM Compatibility**: Pure F32 weights, no SIMD requirements
//! 2. **Fair Comparison**: Same inference algorithm as GGUFTransformer
//! 3. **Serialization**: APR format with model type `TransformerLM` (0x0050)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr_transformer::AprTransformer;
//! use realizar::gguf::{GGUFModel, GGUFTransformer};
//!
//! // Load GGUF model
//! let gguf_data = std::fs::read("model.gguf")?;
//! let gguf_model = GGUFModel::from_bytes(&gguf_data)?;
//! let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data)?;
//!
//! // Convert to APR format
//! let apr_transformer = AprTransformer::from_gguf_transformer(&gguf_transformer);
//!
//! // Run inference (should match GGUF output)
//! let logits = apr_transformer.forward(&[1, 2, 3, 4])?;
//! ```

use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

// PMAT-802: Extracted modules
mod config;
mod convert;
mod dequant;
mod generation;
mod helpers;
mod loader;
mod q4_simd;
pub use config::{
    AprKVCache, AprTransformerConfig, AprTransformerLayer, GenerateConfig, Q4KLayerWeights,
};
use dequant::{dequantize_q4_k_apr, dequantize_q6_k_apr, dequantize_q8_0_apr, f16_to_f32};
use helpers::{matmul_q4k_rowmajor, matmul_q6k_rowmajor, simd_add_weighted, simd_dot_f32};
pub use loader::{
    AprQuantizationType, MmapAprTransformer, QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};
pub use q4_simd::{
    AprInferenceScratch, QuantizedAprLayerQ4, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};

// APR Benchmark Infrastructure (Y6) - extracted from mod.rs (PMAT-802)
mod benchmark;
pub use benchmark::{
    AprBenchmarkResult, AprBenchmarkRunner, AprLoadResult, AprParityComparison, AprPrefillResult,
    APR_CPU_DECODE_THRESHOLD_TOK_S, APR_PARITY_THRESHOLD_PCT, APR_PREFILL_THRESHOLD_TOK_S,
};

// GH-202 FIX: Per-row dequantization for padded quantized matrices.
// quantize_q{4,6}_k_matrix pads each row to 256-element boundary.
// Flat dequant reads blocks sequentially and corrupts data when cols % 256 != 0.
// This function dequantizes per-row, skipping padding at the end of each row.
fn dequant_perrow(
    data: &[u8],
    dims: &[usize],
    block_elems: usize,
    block_bytes: usize,
    dequant_block: impl Fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    let rows = dims[0];
    let cols = dims[1];
    let blocks_per_row = cols.div_ceil(block_elems);
    let bytes_per_row = blocks_per_row * block_bytes;
    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let row_start = row * bytes_per_row;
        if row_start + bytes_per_row > data.len() {
            // Not enough data, fill remaining with zeros
            result.resize(rows * cols, 0.0);
            return result;
        }
        // Dequantize all blocks in this row
        let mut row_values = vec![0.0f32; blocks_per_row * block_elems];
        for b in 0..blocks_per_row {
            let block_start = row_start + b * block_bytes;
            let out_start = b * block_elems;
            dequant_block(
                &data[block_start..block_start + block_bytes],
                &mut row_values[out_start..out_start + block_elems],
            );
        }
        // Keep only the actual cols (discard padding)
        result.extend_from_slice(&row_values[..cols]);
    }
    result
}

// Dequantize a single Q6K super-block (210 bytes → 256 f32 values)
fn dequant_q6k_block(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let mut scales = [0i8; 16];
    #[allow(clippy::cast_possible_wrap)]
    for (i, s) in scales.iter_mut().enumerate() {
        *s = block[192 + i] as i8;
    }
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for n in (0..256).step_by(128) {
        let idx = n / 128;
        let sc = &scales[8 * idx..];
        let ql_s = &ql[64 * idx..];
        let qh_s = &qh[32 * idx..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql_s[l] & 0xF) | ((qh_s[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql_s[l + 32] & 0xF) | (((qh_s[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql_s[l] >> 4) | (((qh_s[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql_s[l + 32] >> 4) | (((qh_s[l] >> 6) & 3) << 4)) as i32 - 32;
            out[n + l] = d * (sc[is] as f32) * (q1 as f32);
            out[n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
            out[n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
            out[n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
        }
    }
}

// Dequantize a single Q4K super-block (144 bytes → 256 f32 values)
// Inlined from dequantize_q4_k_apr single-block logic.
fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    // d (f16) at bytes 0-1, dmin (f16) at bytes 2-3
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    // scales: 12 bytes at offset 4
    let scales = &block[4..16];
    // qs: 128 bytes at offset 16
    let qs = &block[16..144];

    let mut ys_index = 0;
    for j in (0..256).step_by(64) {
        let q = &qs[j / 2..j / 2 + 32];
        let is = j / 32;
        let (sc1, m1) = dequant::extract_scale_min_apr(scales, is);
        let d1 = d * sc1;
        let dm1 = dmin * m1;
        let (sc2, m2) = dequant::extract_scale_min_apr(scales, is + 1);
        let d2 = d * sc2;
        let dm2 = dmin * m2;
        // 32 low nibbles
        for &byte in q {
            out[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
            ys_index += 1;
        }
        // 32 high nibbles
        for &byte in q {
            out[ys_index] = d2 * (byte >> 4) as f32 - dm2;
            ys_index += 1;
        }
    }
}

/// Statistics for a vector of activations
#[derive(Debug, Clone, Default)]
pub struct ActivationStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zeros
    pub zero_count: usize,
    /// Total number of elements
    pub count: usize,
}

impl ActivationStats {
    /// Compute statistics from a slice of floats
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let count = data.len();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = v as f64 - mean as f64;
                var_sum += diff * diff;
            }
        }
        let std_dev = if valid_count > 1 {
            ((var_sum / (valid_count - 1) as f64).sqrt()) as f32
        } else {
            0.0
        };

        Self {
            min,
            max,
            mean,
            std_dev,
            nan_count,
            inf_count,
            zero_count,
            count,
        }
    }
}

/// Per-layer activation trace
#[derive(Debug, Clone)]
pub struct LayerActivation {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Statistics after attention layer norm
    pub attn_norm_stats: ActivationStats,
    /// Statistics after QKV projection
    pub qkv_stats: ActivationStats,
    /// Statistics after attention output
    pub attn_out_stats: ActivationStats,
    /// Statistics after FFN layer norm
    pub ffn_norm_stats: ActivationStats,
    /// Statistics after FFN output
    pub ffn_out_stats: ActivationStats,
    /// Statistics after residual connection (layer output)
    pub output_stats: ActivationStats,
}

/// Forward pass trace with layer-by-layer activations
#[derive(Debug, Clone)]
pub struct ForwardTrace {
    /// Input token IDs
    pub input_tokens: Vec<u32>,
    /// Embedding statistics
    pub embed_stats: ActivationStats,
    /// Per-layer activations
    pub layer_activations: Vec<LayerActivation>,
    /// Final layer norm statistics
    pub final_norm_stats: ActivationStats,
    /// Output logits statistics
    pub logits_stats: ActivationStats,
    /// Final logits vector (for top-k analysis)
    pub logits: Vec<f32>,
}

/// PMAT-216: Trait for inference backends that support layer-by-layer tracing
///
/// This trait ensures all inference backends (CPU, GPU, etc.) provide consistent
/// tracing capability for diagnostics and parity testing.
///
/// # Five Whys Root Cause
///
/// The GPU path lacked tracing while CPU had it, allowing bugs to ship undetected.
/// This trait enforces that any new backend MUST implement tracing from day one.
///
/// # Example
///
/// ```rust,ignore
/// // All backends must implement this:
/// impl TracedForward for MyNewBackend {
///     fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
///         // Must capture layer-by-layer statistics
///     }
/// }
/// ```
pub trait TracedForward {
    /// Run forward pass with layer-by-layer activation statistics
    ///
    /// Returns a `ForwardTrace` containing:
    /// - Embedding statistics
    /// - Per-layer activation statistics (attn_norm, qkv, attn_out, ffn_norm, ffn_out, output)
    /// - Final norm statistics
    /// - Logits statistics and values
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace>;
}

/// APR Transformer model with all weights
///
/// For Q4K models, raw Q4K bytes can be stored in `q4k_layers` to enable
/// fused kernel inference (F-GPU-130) without full dequantization overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformer {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding weights [vocab_size * hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<AprTransformerLayer>,
    /// Output norm weight [hidden_dim]
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional) [hidden_dim]
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [hidden_dim * vocab_size]
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional) [vocab_size]
    pub lm_head_bias: Option<Vec<f32>>,
    /// Q4K raw layer weights for fused kernel inference (F-GPU-130)
    /// When present, enables direct Q4K matmul without dequantization
    #[serde(default)]
    pub q4k_layers: Option<Vec<Q4KLayerWeights>>,
    /// LM head weight in Q6K format for fused kernel inference
    /// When present, enables direct Q6K matmul without dequantization
    #[serde(default)]
    pub lm_head_weight_q6k: Option<Vec<u8>>,
    /// LM head weight in Q4K format for fused kernel inference
    #[serde(default)]
    pub lm_head_weight_q4k: Option<Vec<u8>>,
}

impl AprTransformer {
    /// Load APR transformer from an APR v2 file
    ///
    /// Parses the APR v2 format (magic "APR2") and extracts transformer weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .apr file
    ///
    /// # Returns
    ///
    /// Loaded transformer ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let transformer = AprTransformer::from_apr_file("model.apr")?;
    /// let logits = transformer.forward(&[1, 2, 3])?;
    /// ```
    pub fn from_apr_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let mut file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open APR file: {e}"),
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to read APR file: {e}"),
            })?;

        Self::from_apr_bytes(&data)
    }

    /// Load APR transformer from bytes
    ///
    /// Parses APR v2 format from memory buffer.
    pub fn from_apr_bytes(data: &[u8]) -> Result<Self> {
        // Check minimum size for header
        if data.len() < 64 {
            return Err(RealizarError::FormatError {
                reason: format!("APR file too small: {} bytes (need 64)", data.len()),
            });
        }

        // Check magic - first 3 bytes must be "APR", 4th byte is version (0, '1', or '2')
        let magic = &data[0..4];
        if magic[0..3] != *b"APR" || (magic[3] != 0 && magic[3] != b'1' && magic[3] != b'2') {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: {:?}, expected APR followed by version byte",
                    String::from_utf8_lossy(magic)
                ),
            });
        }

        // Parse header
        // APR header layout:
        //   0-3: Magic "APR\0"
        //   4-5: Version major.minor
        //   6-7: Flags
        //   8-11: Tensor count
        //   12-19: Metadata offset
        //   20-23: Metadata size
        //   24-31: Tensor index offset
        //   32-39: Data offset
        //   40-43: Checksum
        //   44-63: Reserved

        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]) as usize;
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]) as usize;
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]) as usize;

        // Parse metadata (JSON)
        let metadata_end = metadata_offset + metadata_size;
        if metadata_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "Metadata extends beyond file".to_string(),
            });
        }

        let metadata_json = &data[metadata_offset..metadata_end];
        let metadata: serde_json::Value = serde_json::from_slice(metadata_json).unwrap_or_default();

        // Extract architecture info from metadata
        let hidden_dim = metadata
            .get("hidden_size")
            .or_else(|| metadata.get("hidden_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(64) as usize;

        let num_layers = metadata
            .get("num_hidden_layers")
            .or_else(|| metadata.get("num_layers"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;

        let num_heads = metadata
            .get("num_attention_heads")
            .or_else(|| metadata.get("num_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(4) as usize;

        let num_kv_heads = metadata
            .get("num_key_value_heads")
            .or_else(|| metadata.get("num_kv_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(num_heads as u64) as usize;

        let vocab_size = metadata
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32000) as usize;

        let intermediate_dim = metadata
            .get("intermediate_size")
            .or_else(|| metadata.get("intermediate_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        let rope_theta = metadata
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10000.0) as f32;

        let rms_norm_eps = metadata
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;

        let max_position = metadata
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2048) as usize;

        // PMAT-125: Extract architecture from metadata (was missing, defaulted to "unknown")
        // Check "architecture" first (APR v2 standard), then "model_type" (fallback)
        let architecture = metadata
            .get("architecture")
            .or_else(|| metadata.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_lowercase)
            .filter(|s| s != "auto" && !s.is_empty()) // "Auto" is not a valid architecture
            .unwrap_or_else(|| "unknown".to_string());

        let config = AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: max_position,
            rope_theta,
            eps: rms_norm_eps,
        };

        if std::env::var("REALIZE_DEBUG").is_ok() {
            eprintln!("[DEBUG] AprTransformerConfig: hidden_dim={}, num_layers={}, num_heads={}, num_kv_heads={}, vocab_size={}, intermediate_dim={}",
                hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_dim);
        }

        // Parse tensor index
        // APR v2 TensorIndexEntry format:
        //   - name_len (2 bytes) + name (variable)
        //   - dtype (1 byte)
        //   - ndim (1 byte) + dims (8 bytes each)
        //   - offset (8 bytes)
        //   - size (8 bytes)
        // Tuple: (offset, size, dims, dtype)
        let mut tensors: std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)> =
            std::collections::BTreeMap::new();

        let mut pos = tensor_index_offset;
        for _ in 0..tensor_count {
            if pos + 4 > data.len() {
                break;
            }

            // Read tensor name length and name
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            if pos + name_len + 18 > data.len() {
                break;
            }

            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            // Read dtype (1 byte)
            let dtype = data[pos];
            pos += 1;

            // Read ndim (1 byte)
            let ndim = data[pos] as usize;
            pos += 1;

            // Read dimensions (8 bytes each)
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                if pos + 8 > data.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                dims.push(dim);
                pos += 8;
            }

            // Read offset (8 bytes)
            if pos + 16 > data.len() {
                break;
            }
            let offset = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            // Read size (8 bytes)
            let size = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            tensors.insert(name, (data_offset + offset, size, dims, dtype));
        }

        // Helper to extract f32 tensor (with Q4_K dequantization support)
        let get_f32_tensor = |name: &str| -> Option<Vec<f32>> {
            tensors.get(name).map(|(offset, size, dims, dtype)| {
                let end = offset + size;
                if end > data.len() {
                    return Vec::new();
                }
                let tensor_data = &data[*offset..end];

                // GH-191 FIX: Match on GGML dtype values written by converter.
                // Converter now writes GGML types directly: 12=Q4_K, 13=Q5_K, 14=Q6_K, 8=Q8_0
                match dtype {
                    // Q4_K (GGML type 12)
                    12 => {
                        // GH-202 FIX: Handle per-row padding for 2D tensors.
                        // quantize_q4_k_matrix pads each row to 256-element boundary.
                        // Flat dequant would read padding as data, corrupting all rows after first.
                        if dims.len() == 2 && dims[1] % 256 != 0 {
                            dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
                                dequant_q4k_block(block, out);
                            })
                        } else {
                            let num_elements: usize = dims.iter().product();
                            dequantize_q4_k_apr(tensor_data, num_elements)
                        }
                    },
                    // Q5_K (GGML type 13) - use Q4_K dequant (compatible layout)
                    13 => {
                        if dims.len() == 2 && dims[1] % 256 != 0 {
                            dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
                                dequant_q4k_block(block, out);
                            })
                        } else {
                            let num_elements: usize = dims.iter().product();
                            dequantize_q4_k_apr(tensor_data, num_elements)
                        }
                    },
                    // Q6_K (GGML type 14)
                    14 => {
                        // GH-202 FIX: Handle per-row padding for 2D tensors.
                        // quantize_q6_k_matrix pads each row to 256-element boundary.
                        if dims.len() == 2 && dims[1] % 256 != 0 {
                            dequant_perrow(tensor_data, dims, 256, 210, |block, out| {
                                dequant_q6k_block(block, out);
                            })
                        } else {
                            let num_elements: usize = dims.iter().product();
                            dequantize_q6_k_apr(tensor_data, num_elements)
                        }
                    },
                    // Q8_0 (GGML type 8): 34 bytes per block (2 f16 scale + 32 i8 quants)
                    8 => {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q8_0_apr(tensor_data, num_elements)
                    },
                    // F16 (GGML type 1): convert f16 to f32
                    1 => tensor_data
                        .chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                    // F32 (dtype=0) or other: interpret as raw F32
                    _ => tensor_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                }
            })
        };

        // PMAT-103 FIX: Helper to get raw Q4K bytes (no dequantization) for fused kernel
        // GH-191 FIX: Use GGML dtype values (converter now writes these directly)
        //   12 = Q4_K, 13 = Q5_K (treated as Q4_K layout)
        let get_q4k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q4_K (GGML 12) or Q5_K (GGML 13)
                if *dtype != 12 && *dtype != 13 {
                    return None;
                }
                let end = offset + size;
                if end > data.len() {
                    return None;
                }
                Some(data[*offset..end].to_vec())
            })
        };

        // PMAT-103 FIX: Also extract Q6K raw bytes for fused kernel
        // GH-191 FIX: Use GGML dtype value 14 = Q6_K
        let get_q6k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q6_K (GGML 14)
                if *dtype != 14 {
                    return None;
                }
                let end = offset + size;
                if end > data.len() {
                    return None;
                }
                Some(data[*offset..end].to_vec())
            })
        };

        // Debug: print available tensor names (only when REALIZE_DEBUG is set)
        let debug_enabled = std::env::var("REALIZE_DEBUG").is_ok();
        if debug_enabled {
            eprintln!("[DEBUG] APR v2 tensor count: {tensor_count}");
            eprintln!("[DEBUG] Available tensor names (first 10):");
            for (i, (name, (offset, size, dims, dtype))) in tensors.iter().enumerate() {
                if i < 10 {
                    eprintln!(
                        "  {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}"
                    );
                }
            }
        }

        // PMAT-086 FIX: Transpose matrix from GGUF [in_dim, out_dim] to matmul [out_dim, in_dim]
        // GGUF/APR stores weights as [rows, cols] = [in_dim, out_dim] for y = x @ W
        // But our matmul expects [out_dim, in_dim] for y = W @ x (row-major GEMV)
        let _transpose_weight = |data: Vec<f32>, rows: usize, cols: usize| -> Vec<f32> {
            let mut transposed = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    // data[r, c] -> transposed[c, r]
                    let src_idx = r * cols + c;
                    let dst_idx = c * rows + r;
                    if src_idx < data.len() && dst_idx < transposed.len() {
                        transposed[dst_idx] = data[src_idx];
                    }
                }
            }
            transposed
        };

        // PMAT-086: Detect if using GGUF naming (output.weight, blk.X) or HF naming (lm_head.weight)
        // GGUF uses [hidden_dim, vocab_size], HF uses [vocab_size, hidden_dim]
        let is_gguf_model =
            tensors.contains_key("output.weight") || tensors.contains_key("blk.0.attn_q.weight");
        if debug_enabled {
            eprintln!("[DEBUG] is_gguf_model={is_gguf_model}");
        }

        // GH-187: Enhanced logging for embedding tensor (ALWAYS log, not just debug)
        // Transposition mismatch is the most common root cause for incorrect output
        let embed_names = [
            "model.embed_tokens.weight",
            "token_embd.weight",
            "tok_embeddings.weight",
        ];
        let mut embed_dims: Option<Vec<usize>> = None;
        for name in &embed_names {
            if let Some((_offset, _size, dims, _dtype)) = tensors.get(*name) {
                embed_dims = Some(dims.clone());
                // ALWAYS log embedding info - this is critical for debugging
                eprintln!(
                    "[APR-LOAD] Embedding tensor '{}': dims={:?}, expected [vocab={}, hidden={}]",
                    name, dims, vocab_size, hidden_dim
                );
                break;
            }
        }

        // Try to load token embedding - FAIL FAST if not found (no silent zeros)
        let token_embedding_raw = get_f32_tensor("model.embed_tokens.weight")
            .or_else(|| get_f32_tensor("token_embd.weight"))
            .or_else(|| get_f32_tensor("tok_embeddings.weight"))
            .ok_or_else(|| RealizarError::FormatError {
                reason: "FATAL: No embedding tensor found. Tried: model.embed_tokens.weight, \
                        token_embd.weight, tok_embeddings.weight. APR file may be corrupt or \
                        use unsupported tensor naming convention.".to_string()
            })?;

        // GH-208 FIX: Do NOT transpose embedding data
        // GGML data layout: data[i0 + i1*ne0] for shape [ne0, ne1]
        // For embedding [ne0=hidden, ne1=vocab]: data[h + v*hidden] = row-major [vocab, hidden]
        // Token v's embedding = data[v*hidden .. (v+1)*hidden] - ALREADY CORRECT
        // The GH-187 transpose was WRONG - it corrupted embeddings (correlation 0.001 instead of 1.0)
        // See: contracts/tensor-layout-v1.yaml, compare_embed example
        let token_embedding = token_embedding_raw;
        if let Some(ref dims) = embed_dims {
            eprintln!(
                "[APR-LOAD] Embedding dims={:?}, using raw data (no transpose needed)",
                dims
            );
        }

        // GH-187: Sanity check - verify embedding produces non-garbage for token 0
        if token_embedding.len() >= hidden_dim {
            let first_embed = &token_embedding[0..hidden_dim];
            let all_zero = first_embed.iter().all(|&x| x == 0.0);
            let has_nan = first_embed.iter().any(|x| x.is_nan());
            let has_inf = first_embed.iter().any(|x| x.is_infinite());
            if all_zero {
                eprintln!(
                    "[APR-LOAD] WARNING: Token 0 embedding is all zeros - possible load failure"
                );
            }
            if has_nan || has_inf {
                eprintln!(
                    "[APR-LOAD] ERROR: Token 0 embedding contains NaN/Inf - data corruption!"
                );
            }
            eprintln!(
                "[APR-LOAD] Token 0 embedding sample: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                first_embed[0],
                first_embed.get(1).unwrap_or(&0.0),
                first_embed.get(2).unwrap_or(&0.0),
                first_embed.get(3).unwrap_or(&0.0),
                first_embed.get(4).unwrap_or(&0.0)
            );
        }

        // Log total embedding size (always, for diagnostics)
        eprintln!(
            "[APR-LOAD] Embedding loaded: {} elements (vocab={} x hidden={})",
            token_embedding.len(),
            vocab_size,
            hidden_dim
        );

        // Load output norm
        let output_norm_weight = get_f32_tensor("model.norm.weight")
            .or_else(|| get_f32_tensor("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden_dim]);

        // GH-187: Enhanced logging for lm_head tensor
        for name in &["lm_head.weight", "output.weight"] {
            if let Some((_offset, _size, dims, dtype)) = tensors.get(*name) {
                eprintln!(
                    "[APR-LOAD] LM head tensor '{}': dims={:?}, dtype={}, expected [vocab={}, hidden={}]",
                    name, dims, dtype, vocab_size, hidden_dim
                );
                break;
            }
        }

        // Load LM head - FAIL FAST if not found (no silent zeros)
        // For tied embeddings (common in Qwen, LLaMA models), use embed_tokens as fallback
        let lm_head_raw =
            get_f32_tensor("lm_head.weight").or_else(|| get_f32_tensor("output.weight"));
        let (lm_head_raw, used_tied_weights) = if let Some(lm_head) = lm_head_raw {
            (lm_head, false)
        } else {
            // Weight tying: use embedding weights for lm_head
            eprintln!("[APR-LOAD] No lm_head found, trying tied embedding weights");
            let tied = get_f32_tensor("model.embed_tokens.weight")
                .or_else(|| get_f32_tensor("token_embd.weight"));
            if let Some(t) = tied {
                (t, true)
            } else {
                return Err(RealizarError::FormatError {
                    reason: "FATAL: No lm_head tensor found and no embedding for weight tying. \
                            Tried: lm_head.weight, output.weight, model.embed_tokens.weight, \
                            token_embd.weight. APR file may be corrupt.".to_string()
                });
            }
        };
        if used_tied_weights {
            eprintln!("[APR-LOAD] Using tied weights: embedding -> lm_head");
        }

        // GH-187: lm_head is used for matmul (not lookup), so GGML layout is correct
        // lm_head: y = x @ W where W is [hidden_dim, vocab_size] in GGML convention
        // This matches fused_q4k_parallel_matvec(weights, x, in_dim=hidden, out_dim=vocab)
        // NO transposition needed for lm_head (unlike embedding)
        let lm_head_weight = lm_head_raw;
        eprintln!(
            "[APR-LOAD] LM head loaded: {} elements (hidden={} x vocab={})",
            lm_head_weight.len(),
            hidden_dim,
            vocab_size
        );

        // PMAT-103: Load lm_head Q4K/Q6K raw bytes for fused kernel inference
        let lm_head_weight_q4k =
            get_q4k_raw_bytes("lm_head.weight").or_else(|| get_q4k_raw_bytes("output.weight"));
        let lm_head_weight_q6k =
            get_q6k_raw_bytes("lm_head.weight").or_else(|| get_q6k_raw_bytes("output.weight"));
        // GH-187: Always log quantization path for lm_head
        if let Some(ref bytes) = lm_head_weight_q4k {
            eprintln!(
                "[APR-LOAD] LM head using Q4K fused kernel ({} bytes)",
                bytes.len()
            );
        } else if let Some(ref bytes) = lm_head_weight_q6k {
            eprintln!(
                "[APR-LOAD] LM head using Q6K fused kernel ({} bytes)",
                bytes.len()
            );
        } else {
            eprintln!("[APR-LOAD] LM head using F32 matmul (no Q4K/Q6K found)");
        }

        // Compute KV dimension from config
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Load layers
        let mut layers = Vec::with_capacity(num_layers);
        // PMAT-103 FIX: Also extract Q4K raw bytes for fused kernel inference
        let mut q4k_layer_weights: Vec<Q4KLayerWeights> = Vec::with_capacity(num_layers);
        let mut has_any_q4k = false;

        for i in 0..num_layers {
            let hf_prefix = format!("model.layers.{i}");
            let gguf_prefix = format!("blk.{i}");

            // Try separate Q/K/V or combined QKV
            // Support both HuggingFace and GGUF naming conventions
            // PMAT-086 FIX: HF uses [out_dim, in_dim], GGUF uses [in_dim, out_dim]
            // Only transpose GGUF tensors, not HF tensors
            let qkv_out_dim = hidden_dim + kv_dim + kv_dim;

            // Detect if using GGUF naming (blk.X) or HF naming (model.layers.X)
            let is_gguf = tensors.contains_key(&format!("{gguf_prefix}.attn_q.weight"));

            let qkv_weight = if let Some(qkv) =
                get_f32_tensor(&format!("{hf_prefix}.self_attn.qkv_proj.weight"))
            {
                // HF fused QKV - already in [qkv_out_dim, hidden_dim] format
                qkv
            } else {
                // Get Q weight
                let q_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.q_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_q.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
                // Get K weight
                let k_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.k_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_k.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
                // Get V weight
                let v_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_v.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);

                // PMAT-086 FIX: Both HF and GGUF data are in [out_dim, in_dim] layout
                // GGUF dims say [in_dim, out_dim] but data is actually [out_dim, in_dim] due to GGML column-major convention
                // Fuse Q, K, V by stacking rows (Q, then K, then V) - no transpose needed
                let _ = is_gguf; // Suppress unused warning
                let mut qkv = Vec::with_capacity(qkv_out_dim * hidden_dim);
                qkv.extend_from_slice(&q_raw);
                qkv.extend_from_slice(&k_raw);
                qkv.extend_from_slice(&v_raw);
                qkv
            };

            // Get Q/K/V biases (optional, for Qwen models)
            // PMAT-114 FIX: Also check for fused QKV bias from APR converter
            let qkv_bias = if let Some(fused_bias) =
                get_f32_tensor(&format!("{hf_prefix}.self_attn.qkv_proj.bias"))
            {
                // Fused QKV bias from APR converter - use directly
                Some(fused_bias)
            } else {
                // Try separate Q/K/V biases
                let q_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.q_proj.bias"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_q.bias")));
                let k_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.k_proj.bias"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_k.bias")));
                let v_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.v_proj.bias"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_v.bias")));

                // Combine biases if present
                match (&q_bias, &k_bias, &v_bias) {
                    (Some(q), Some(k), Some(v)) => {
                        let mut bias = Vec::with_capacity(qkv_out_dim);
                        bias.extend_from_slice(q);
                        bias.extend_from_slice(k);
                        bias.extend_from_slice(v);
                        Some(bias)
                    },
                    _ => None,
                }
            };

            // PMAT-086 FIX: Both HF and GGUF data are in [out_dim, in_dim] layout - no transpose needed
            let attn_output = get_f32_tensor(&format!("{hf_prefix}.self_attn.o_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_output.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);

            let attn_norm = get_f32_tensor(&format!("{hf_prefix}.input_layernorm.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_norm.weight")))
                .unwrap_or_else(|| vec![1.0; hidden_dim]);

            let ffn_norm = get_f32_tensor(&format!("{hf_prefix}.post_attention_layernorm.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_norm.weight")));

            // PMAT-086 FIX: FFN weights - both HF and GGUF data are in [out_dim, in_dim] layout
            // No transpose needed - GGML column-major dims but row-major data
            let ffn_gate = get_f32_tensor(&format!("{hf_prefix}.mlp.gate_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_gate.weight")));
            let ffn_up = get_f32_tensor(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_up.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = get_f32_tensor(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_down.weight")))
                .unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            // PMAT-103 FIX: Extract Q4K and Q6K raw bytes for fused kernel
            // LAYOUT-002 FIX: Check BOTH naming conventions (HF first, GGUF fallback)
            // Toyota Way: ONE consistent pattern for all tensor lookups (matches F32 path)
            let q4k_attn_q = get_q4k_raw_bytes(&format!("{hf_prefix}.self_attn.q_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_q.weight")));
            let q4k_attn_k = get_q4k_raw_bytes(&format!("{hf_prefix}.self_attn.k_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_k.weight")));
            let q4k_attn_v = get_q4k_raw_bytes(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_v.weight")));
            let q6k_attn_v = get_q6k_raw_bytes(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                .or_else(|| get_q6k_raw_bytes(&format!("{gguf_prefix}.attn_v.weight")));
            let q4k_attn_output = get_q4k_raw_bytes(&format!("{hf_prefix}.self_attn.o_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_output.weight")));
            let q4k_ffn_gate = get_q4k_raw_bytes(&format!("{hf_prefix}.mlp.gate_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_gate.weight")));
            let q4k_ffn_up = get_q4k_raw_bytes(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight")));
            let q4k_ffn_down = get_q4k_raw_bytes(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight")));
            // Q6K fallback for tensors that aren't Q4K (common in mixed quantization models)
            let q6k_ffn_down = get_q6k_raw_bytes(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight")));
            let q6k_ffn_up = get_q6k_raw_bytes(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight")));

            let has_q4k_weights = q4k_attn_q.is_some()
                || q4k_attn_k.is_some()
                || q4k_attn_output.is_some()
                || q4k_ffn_gate.is_some()
                || q4k_ffn_up.is_some()
                || q4k_ffn_down.is_some();
            let has_q6k_weights =
                q6k_ffn_down.is_some() || q6k_ffn_up.is_some() || q6k_attn_v.is_some();

            if has_q4k_weights || has_q6k_weights {
                has_any_q4k = true;
            }

            q4k_layer_weights.push(Q4KLayerWeights {
                qkv_weight: None, // Q+K+V are separate tensors, not combined
                attn_q_weight: q4k_attn_q,
                attn_k_weight: q4k_attn_k,
                attn_v_weight: q4k_attn_v,
                attn_v_weight_q6k: q6k_attn_v,
                attn_output_weight: q4k_attn_output,
                ffn_gate_weight: q4k_ffn_gate,
                ffn_up_weight: q4k_ffn_up,
                ffn_down_weight: q4k_ffn_down,
                ffn_down_weight_q6k: q6k_ffn_down,
                ffn_up_weight_q6k: q6k_ffn_up,
            });

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: attn_output,
                attn_output_bias: None,
                ffn_gate_weight: ffn_gate,
                ffn_gate_bias: None,
                ffn_up_weight: ffn_up,
                ffn_up_bias: None,
                ffn_down_weight: ffn_down,
                ffn_down_bias: None,
                ffn_norm_weight: ffn_norm,
                ffn_norm_bias: None,
            });
        }

        // PMAT-103 FIX: Store Q4K layer weights for fused kernel inference
        let q4k_layers = if has_any_q4k {
            if debug_enabled {
                eprintln!("[DEBUG] Loaded Q4K raw bytes for fused kernel inference");
            }
            Some(q4k_layer_weights)
        } else {
            None
        };

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers,
            lm_head_weight_q6k,
            lm_head_weight_q4k,
        })
    }

    /// Create a new APR transformer with the given configuration
    pub fn new(config: AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;

        let layers = (0..config.num_layers)
            .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
            .collect();

        Self {
            config,
            token_embedding: vec![0.0; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Generate tokens autoregressively (simplified version without KV cache)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;

            // Greedy sampling: take argmax
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            tokens.push(next_token);

            // Stop at EOS tokens:
            // - Standard: 2
            // - Qwen2: 151645 (EOS), 151643 (BOS)
            // - LLaMA: 2
            if next_token == 2 || next_token == 151645 || next_token == 151643 {
                break;
            }
        }

        Ok(tokens)
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        for layer in &self.layers {
            count += layer.num_parameters();
        }
        count += self.output_norm_weight.len();
        count += self.output_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.lm_head_weight.len();
        count += self.lm_head_bias.as_ref().map_or(0, Vec::len);
        count
    }

    /// Get memory size in bytes (F32 = 4 bytes per param)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.num_parameters() * 4
    }

    /// Look up token embeddings
    #[must_use]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let debug = std::env::var("REALIZE_DEBUG").is_ok();
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                if debug && token_id < 10 {
                    eprintln!(
                        "[DEBUG] embed token {}: offset={}, first 5: {:?}",
                        token_id,
                        offset,
                        &self.token_embedding[offset..offset + 5.min(hidden_dim)]
                    );
                }
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // Out of vocab - return zeros
                if debug {
                    eprintln!(
                        "[DEBUG] embed token {}: OUT OF VOCAB (offset {} > {})",
                        token_id,
                        offset,
                        self.token_embedding.len()
                    );
                }
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// RMSNorm (delegates to helpers module)
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        helpers::rms_norm(input, weight, bias, self.config.hidden_dim, eps)
    }

    /// Matrix multiplication (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        helpers::f32_matmul(input, weight, in_dim, out_dim)
    }

    /// Add bias in-place (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        helpers::add_bias_inplace(data, bias);
    }

    /// GELU activation (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        helpers::gelu_inplace(data);
    }

    /// Apply RoPE (delegates to helpers module)
    fn apply_rope_f32(&self, x: &mut [f32], position: usize, num_heads: usize, head_dim: usize) {
        helpers::apply_rope_f32(x, position, num_heads, head_dim, self.config.rope_theta);
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        // NOISY-GUARD: Only print trace messages when REALIZE_TRACE is set
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available)
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            // Calculate qkv_dim from actual weight size (handles GQA models)
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Proper attention with GQA support and RoPE
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Split QKV and apply RoPE
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V (layout: [Q..., K..., V...])
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // Apply RoPE to Q and K
                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Compute scaled dot-product attention with causal mask
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..seq_len {
                    // Compute attention scores for this position
                    let mut scores = Vec::with_capacity(i + 1);
                    let q_start = i * hidden_dim + q_head_offset;

                    for j in 0..=i {
                        // Only attend to positions <= current (causal mask)
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_all[q_start + d] * k_all[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    if exp_sum > 0.0 {
                        for s in &mut scores {
                            *s /= exp_sum;
                        }
                    }

                    // Weighted sum of V
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &weight) in scores.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += weight * v_all[v_start + d];
                        }
                    }
                }
            }

            // 2d. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available
            let mut attn_output = if let Some(q4k_bytes) =
                q4k_layer.and_then(|q| q.attn_output_weight.as_ref())
            {
                if trace_enabled && layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using Q4K fused kernel");
                }
                // Fused Q4K matmul: process each position separately
                // PMAT-103: Use column-major kernel for GGUF layout
                let seq_len = token_ids.len();
                let mut output = Vec::with_capacity(seq_len * hidden_dim);
                for s in 0..seq_len {
                    let input_slice = &attn_out[s * hidden_dim..(s + 1) * hidden_dim];
                    let pos_out =
                        matmul_q4k_rowmajor(q4k_bytes, input_slice, hidden_dim, hidden_dim)?;
                    output.extend(pos_out);
                }
                output
            } else {
                if trace_enabled && layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using F32 fallback (slow!)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2g. FFN projection (SwiGLU or standard GELU)
            // PMAT-103: Use Q4K fused kernel when available for FFN
            let seq_len = token_ids.len();
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // PMAT-103: Check for Q4K gate weight
                let gate =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 {
                            eprintln!("[TRACE] Layer {layer_idx}: ffn_gate using Q4K fused kernel");
                        }
                        let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                        for s in 0..seq_len {
                            let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                            // ffn_gate: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                            let pos_out = matmul_q4k_rowmajor(
                                q4k_bytes,
                                input_slice,
                                intermediate_dim,
                                hidden_dim,
                            )?;
                            output.extend(pos_out);
                        }
                        output
                    } else {
                        // AUDIT-301: Use already-bound _gate_weight instead of expect()
                        self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim)
                    };

                // PMAT-103: Check for Q4K up weight
                let up = if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_up using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                        // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                        let pos_out = matmul_q4k_rowmajor(
                            q4k_bytes,
                            input_slice,
                            intermediate_dim,
                            hidden_dim,
                        )?;
                        output.extend(pos_out);
                    }
                    output
                } else {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_up using F32 fallback (slow!)");
                    }
                    self.matmul(
                        &ffn_input,
                        &layer.ffn_up_weight,
                        hidden_dim,
                        intermediate_dim,
                    )
                };

                // SiLU(gate) * up, then down projection
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp()); // SiLU = x * sigmoid(x)
                    ffn_hidden.push(silu_g * u);
                }

                // PMAT-103: Check for Q4K or Q6K down weight
                let mut out = if let Some(q4k_bytes) =
                    q4k_layer.and_then(|q| q.ffn_down_weight.as_ref())
                {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice =
                            &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q4k_rowmajor(
                            q4k_bytes,
                            input_slice,
                            hidden_dim,
                            intermediate_dim,
                        )?;
                        output.extend(pos_out);
                    }
                    output
                } else if let Some(q6k_bytes) =
                    q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref())
                {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q6K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice =
                            &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q6k_rowmajor(
                            q6k_bytes,
                            input_slice,
                            hidden_dim,
                            intermediate_dim,
                        )?;
                        output.extend(pos_out);
                    }
                    output
                } else {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using F32 fallback (slow!)");
                    }
                    self.matmul(
                        &ffn_hidden,
                        &layer.ffn_down_weight,
                        intermediate_dim,
                        hidden_dim,
                    )
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP: down(GELU(up(x)))
                // PMAT-103: Check for Q4K up weight
                let mut ffn_hidden =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                        let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                        for s in 0..seq_len {
                            let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                            // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                            let pos_out = matmul_q4k_rowmajor(
                                q4k_bytes,
                                input_slice,
                                intermediate_dim,
                                hidden_dim,
                            )?;
                            output.extend(pos_out);
                        }
                        output
                    } else {
                        self.matmul(
                            &ffn_input,
                            &layer.ffn_up_weight,
                            hidden_dim,
                            intermediate_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                self.gelu(&mut ffn_hidden);

                // PMAT-103: Check for Q4K down weight
                let mut out =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                        let mut output = Vec::with_capacity(seq_len * hidden_dim);
                        for s in 0..seq_len {
                            let input_slice =
                                &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                            let pos_out = matmul_q4k_rowmajor(
                                q4k_bytes,
                                input_slice,
                                hidden_dim,
                                intermediate_dim,
                            )?;
                            output.extend(pos_out);
                        }
                        output
                    } else {
                        self.matmul(
                            &ffn_hidden,
                            &layer.ffn_down_weight,
                            intermediate_dim,
                            hidden_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };

            // 2h. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Forward pass with layer-by-layer activation tracing.
    ///
    /// This is identical to `forward()` but collects statistics at each layer
    /// for debugging inference divergence issues.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// `ForwardTrace` containing logits and per-layer activation statistics
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward_traced(&self, token_ids: &[u32]) -> Result<ForwardTrace> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);
        let embed_stats = ActivationStats::from_slice(&hidden);

        let mut layer_activations = Vec::with_capacity(self.layers.len());

        // 2. Process through transformer layers with tracing
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Note: Q4K layers not used in traced forward (uses F32 for accuracy)
            let _q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );
            let attn_norm_stats = ActivationStats::from_slice(&normed);

            // 2b. QKV projection
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }
            let qkv_stats = ActivationStats::from_slice(&qkv);

            // 2c. Attention computation (simplified for trace - same logic as forward)
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Attention output
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..seq_len {
                    let mut scores = Vec::with_capacity(i + 1);
                    let q_start = i * hidden_dim + q_head_offset;

                    for j in 0..=i {
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_all[q_start + d] * k_all[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                    // Weighted sum of values
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &p) in probs.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += p * v_all[v_start + d];
                        }
                    }
                }
            }

            // Output projection
            let mut attn_output = self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }
            let attn_out_stats = ActivationStats::from_slice(&attn_output);

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN layer norm (if present)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                let normed = self.layer_norm(
                    &hidden,
                    norm_weight,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                );
                normed
            } else {
                hidden.clone()
            };
            let ffn_norm_stats = ActivationStats::from_slice(&ffn_input);

            // 2g. FFN - check if gated MLP (SwiGLU) by presence of gate weight
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let gate = self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim);
                let up = self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim);

                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp());
                    ffn_hidden.push(silu_g * u);
                }

                let mut out = self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim);
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP without gating
                let mut ffn_hidden = self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim);
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                for h in &mut ffn_hidden {
                    let gelu_approx = 0.5 * *h * (1.0 + (0.797_884_6 * (*h + 0.044_715 * *h * *h * *h)).tanh());
                    *h = gelu_approx;
                }
                let mut out = self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim);
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };
            let ffn_out_stats = ActivationStats::from_slice(&ffn_output);

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
            let output_stats = ActivationStats::from_slice(&hidden);

            layer_activations.push(LayerActivation {
                layer_idx,
                attn_norm_stats,
                qkv_stats,
                attn_out_stats,
                ffn_norm_stats,
                ffn_out_stats,
                output_stats,
            });
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );
        let final_norm_stats = ActivationStats::from_slice(&normed);

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }
        let logits_stats = ActivationStats::from_slice(&logits);

        Ok(ForwardTrace {
            input_tokens: token_ids.to_vec(),
            embed_stats,
            layer_activations,
            final_norm_stats,
            logits_stats,
            logits,
        })
    }

    /// Predict next token (greedy decoding)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Token ID with highest probability
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;

        // Argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;

        Ok(max_idx as u32)
    }

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `cache` - Mutable KV cache to read from and append to
    /// * `position` - Position in sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    pub fn forward_with_cache(
        &self,
        token_id: u32,
        cache: &mut AprKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        // DEBUG: Force F32 fallback to verify data layout issues
        let force_f32 = std::env::var("APR_FORCE_F32").is_ok();
        // NOISY-GUARD: Only print trace messages when REALIZE_TRACE is set
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // 2. Process through transformer layers
        let layers_start = std::time::Instant::now();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available) for fused kernels
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // F-REGR-231: Debug normed input
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: normed[0..5] = {:?}",
                    &normed[..5.min(normed.len())]
                );
            }

            // 2b. QKV projection (single token)
            // PMAT-103: Use fused Q4K kernels for separate Q, K, V weights when available
            let kv_size = num_kv_heads * head_dim;
            let (mut q, mut k, v) = if let Some(q4k) = q4k_layer {
                // Try Q4K fused kernels for Q, K
                let q = if let Some(ref q_bytes) = q4k.attn_q_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: Q projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q_bytes, &normed, hidden_dim, hidden_dim)?
                } else {
                    // Fallback to F32 for Q (should not happen for GGUF models)
                    let q_weight = &layer.qkv_weight[0..hidden_dim * hidden_dim];
                    self.matmul(&normed, q_weight, hidden_dim, hidden_dim)
                };

                let k = if let Some(ref k_bytes) = q4k.attn_k_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: K projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(k_bytes, &normed, kv_size, hidden_dim)?
                } else {
                    let k_start = hidden_dim * hidden_dim;
                    let k_weight = &layer.qkv_weight[k_start..k_start + kv_size * hidden_dim];
                    self.matmul(&normed, k_weight, hidden_dim, kv_size)
                };

                // V can be Q4K or Q6K
                let v = if let Some(ref v_bytes) = q4k.attn_v_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: V projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)?
                } else if let Some(ref v_bytes) = q4k.attn_v_weight_q6k {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: V projection using Q6K fused kernel");
                    }
                    matmul_q6k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)?
                } else {
                    let v_start = hidden_dim * hidden_dim + kv_size * hidden_dim;
                    let v_weight = &layer.qkv_weight[v_start..v_start + kv_size * hidden_dim];
                    self.matmul(&normed, v_weight, hidden_dim, kv_size)
                };

                (q, k, v)
            } else {
                // Fallback: Combined QKV with F32 (legacy path)
                if trace_enabled && layer_idx == 0 && position == 0 {
                    eprintln!("[TRACE-CACHE] Layer 0: QKV projection using F32 (not fused)");
                }
                let qkv_out_dim = layer.qkv_weight.len() / hidden_dim;
                let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_out_dim);
                if let Some(ref bias) = layer.qkv_bias {
                    self.add_bias(&mut qkv, bias);
                }
                let q = qkv[0..hidden_dim].to_vec();
                let k = qkv[hidden_dim..hidden_dim + kv_size].to_vec();
                let v = qkv[hidden_dim + kv_size..hidden_dim + 2 * kv_size].to_vec();
                (q, k, v)
            };

            // Apply biases if present (for fused path)
            // The combined qkv_bias is [Q_bias | K_bias | V_bias]
            let mut v_mut = v;
            if q4k_layer.is_some() {
                if let Some(ref bias) = layer.qkv_bias {
                    // Split bias into Q, K, V portions
                    for (i, b) in bias[0..hidden_dim].iter().enumerate() {
                        q[i] += b;
                    }
                    for (i, b) in bias[hidden_dim..hidden_dim + kv_size].iter().enumerate() {
                        k[i] += b;
                    }
                    // V bias starts after Q and K biases
                    let v_bias_start = hidden_dim + kv_size;
                    for (i, b) in bias[v_bias_start..v_bias_start + kv_size]
                        .iter()
                        .enumerate()
                    {
                        v_mut[i] += b;
                    }
                }
            }
            let v = v_mut;

            // F-REGR-231: Debug K after bias
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: K after bias[0..5] = {:?}",
                    &k[..5.min(k.len())]
                );
                eprintln!(
                    "[TRACE-CACHE] Layer 0: V after bias[0..5] = {:?}",
                    &v[..5.min(v.len())]
                );
            }

            // PMAT-103: Apply RoPE to Q and K at current position
            // This was missing, causing garbage output
            self.apply_rope_f32(&mut q, position, num_heads, head_dim);
            self.apply_rope_f32(&mut k, position, num_kv_heads, head_dim);

            // F-REGR-231: Debug K after RoPE
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: K after RoPE[0..5] = {:?}",
                    &k[..5.min(k.len())]
                );
            }

            // 2c. Append K, V to cache (K now has RoPE applied)
            cache.append(layer_idx, &k, &v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let cache_len = cache.len();

            // F-REGR-231 FIX: Handle first token specially
            // When cache is empty (cache_len = 0), we just appended K/V but len isn't incremented yet.
            // For the first token, attention(single token) = V directly (since softmax of one score = 1.0).
            let attn_out = if cache_len == 0 {
                // First token: just use V directly (expanded via GQA)
                let group_size = num_heads / num_kv_heads;
                (0..num_heads)
                    .flat_map(|h| {
                        let kv_head = h / group_size;
                        let start = kv_head * head_dim;
                        v[start..start + head_dim].iter().copied()
                    })
                    .collect()
            } else {
                // Subsequent tokens: use full attention with cache + current K/V
                let mut attn_out = vec![0.0f32; hidden_dim];
                let scale = 1.0 / (head_dim as f32).sqrt();

                // seq_len includes cached positions plus current position
                let seq_len = cache_len + 1;

                for h in 0..num_heads {
                    let kv_head = h * num_kv_heads / num_heads; // GQA mapping
                    let q_start = h * head_dim;
                    let q_slice = &q[q_start..q_start + head_dim];

                    // Compute attention scores with SIMD dot product
                    let mut scores = Vec::with_capacity(seq_len);

                    // Scores for cached positions
                    for pos in 0..cache_len {
                        let k_start = pos * kv_size + kv_head * head_dim;
                        let k_slice = &k_cache[k_start..k_start + head_dim];
                        let dot = simd_dot_f32(q_slice, k_slice);
                        scores.push(dot * scale);
                    }

                    // Score for current position (using current K)
                    let k_start = kv_head * head_dim;
                    let k_slice = &k[k_start..k_start + head_dim];
                    let dot = simd_dot_f32(q_slice, k_slice);
                    scores.push(dot * scale);

                    // Causal mask: only attend to positions <= current
                    for pos in (position + 1)..seq_len {
                        scores[pos] = f32::NEG_INFINITY;
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum: f32 = exp_scores.iter().sum();
                    if sum > 0.0 {
                        let inv_sum = 1.0 / sum;
                        for s in &mut exp_scores {
                            *s *= inv_sum;
                        }
                    }

                    // Weighted sum of V
                    let attn_out_head = &mut attn_out[q_start..q_start + head_dim];

                    // From cached positions
                    for pos in 0..cache_len {
                        let v_start = pos * kv_size + kv_head * head_dim;
                        let v_slice = &v_cache[v_start..v_start + head_dim];
                        simd_add_weighted(attn_out_head, v_slice, exp_scores[pos]);
                    }

                    // From current position (using current V)
                    let v_start = kv_head * head_dim;
                    let v_slice = &v[v_start..v_start + head_dim];
                    simd_add_weighted(attn_out_head, v_slice, exp_scores[cache_len]);
                }

                attn_out
            };

            // F-REGR-231: Debug attn_out before projection
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: attn_out[0..5] = {:?} (before output projection)",
                    &attn_out[..5.min(attn_out.len())]
                );
                let attn_out_sum: f32 = attn_out.iter().sum();
                eprintln!("[TRACE-CACHE] Layer 0: attn_out sum = {:.4}", attn_out_sum);
            }

            // 2e. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available (single token path)
            let mut attn_output = if !force_f32 {
                if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.attn_output_weight.as_ref()) {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q4k_bytes, &attn_out, hidden_dim, hidden_dim)?
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 fallback (slow!)");
                    }
                    self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
                }
            } else {
                if trace_enabled && layer_idx == 0 && position == 0 {
                    eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 (APR_FORCE_F32)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // F-REGR-231: Debug attn_output after projection
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: attn_output[0..5] = {:?} (after output projection)",
                    &attn_output[..5.min(attn_output.len())]
                );
            }

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // F-REGR-231: Debug hidden after attention
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: hidden_after_attn[0..5] = {:?}",
                    &hidden[..5.min(hidden.len())]
                );
            }

            // 2g. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2h. FFN projection (SwiGLU or standard GELU)
            // PMAT-103 FIX: Use Q4K/Q6K fused kernels when available (single token path)
            let intermediate_dim = self.config.intermediate_dim;
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // PMAT-103: Check for Q4K gate weight
                let gate = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)?
                    } else {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 fallback (slow!)");
                        }
                        // AUDIT-301: Use already-bound _gate_weight instead of expect()
                        self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim)
                    }
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 (APR_FORCE_F32)");
                    }
                    // AUDIT-301: Use already-bound _gate_weight instead of expect()
                    self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim)
                };

                // PMAT-103: Check for Q4K/Q6K up weight
                let up = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)?
                    } else if let Some(q6k_bytes) =
                        q4k_layer.and_then(|q| q.ffn_up_weight_q6k.as_ref())
                    {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q6K fused kernel");
                        }
                        matmul_q6k_rowmajor(q6k_bytes, &ffn_input, intermediate_dim, hidden_dim)?
                    } else {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 fallback (slow!)");
                        }
                        self.matmul(
                            &ffn_input,
                            &layer.ffn_up_weight,
                            hidden_dim,
                            intermediate_dim,
                        )
                    }
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(
                        &ffn_input,
                        &layer.ffn_up_weight,
                        hidden_dim,
                        intermediate_dim,
                    )
                };

                // SiLU(gate) * up, then down projection
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp()); // SiLU = x * sigmoid(x)
                    ffn_hidden.push(silu_g * u);
                }

                // PMAT-103: Check for Q4K or Q6K down weight
                let mut out = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else if let Some(q6k_bytes) =
                        q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref())
                    {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q6K fused kernel");
                        }
                        matmul_q6k_rowmajor(q6k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 fallback (slow!)");
                        }
                        self.matmul(
                            &ffn_hidden,
                            &layer.ffn_down_weight,
                            intermediate_dim,
                            hidden_dim,
                        )
                    }
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(
                        &ffn_hidden,
                        &layer.ffn_down_weight,
                        intermediate_dim,
                        hidden_dim,
                    )
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP: down(GELU(up(x)))
                // PMAT-103: Check for Q4K up weight
                let mut ffn_hidden =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)?
                    } else {
                        self.matmul(
                            &ffn_input,
                            &layer.ffn_up_weight,
                            hidden_dim,
                            intermediate_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                self.gelu(&mut ffn_hidden);

                // PMAT-103: Check for Q4K down weight
                let mut out =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else {
                        self.matmul(
                            &ffn_hidden,
                            &layer.ffn_down_weight,
                            intermediate_dim,
                            hidden_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };

            // 2i. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            // F-REGR-231: Debug hidden state after each layer
            if trace_enabled && layer_idx < 2 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] After layer {}: hidden[0..5] = {:?}",
                    layer_idx,
                    &hidden[..5.min(hidden.len())]
                );
            }
        }
        if trace_enabled {
            eprintln!(
                "[TRACE-CACHE] pos={}: {} layers took {:?}",
                position,
                self.layers.len(),
                layers_start.elapsed()
            );
        }

        // NOTE: No advance() needed here - append() auto-advances on the last layer
        // (see F-REGR-231 fix in config.rs)

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        // PMAT-103: Use Q4K/Q6K fused kernel when available (single token path)
        let lm_start = std::time::Instant::now();
        let mut logits = if !force_f32 {
            if let Some(ref q4k_bytes) = self.lm_head_weight_q4k {
                if trace_enabled {
                    eprintln!("[TRACE-CACHE] lm_head using Q4K fused kernel");
                }
                matmul_q4k_rowmajor(q4k_bytes, &normed, self.config.vocab_size, hidden_dim)?
            } else if let Some(ref q6k_bytes) = self.lm_head_weight_q6k {
                let result =
                    matmul_q6k_rowmajor(q6k_bytes, &normed, self.config.vocab_size, hidden_dim)?;
                if trace_enabled {
                    eprintln!("[TRACE-CACHE] lm_head Q6K took {:?}", lm_start.elapsed());
                }
                result
            } else {
                self.matmul(
                    &normed,
                    &self.lm_head_weight,
                    hidden_dim,
                    self.config.vocab_size,
                )
            }
        } else {
            if trace_enabled {
                eprintln!("[TRACE-CACHE] lm_head using F32 (APR_FORCE_F32)");
            }
            self.matmul(
                &normed,
                &self.lm_head_weight,
                hidden_dim,
                self.config.vocab_size,
            )
        };
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens using KV cache (delegates to generation module)
    pub fn generate_with_cache(&self, prompt: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        generation::generate_with_cache(self, prompt, config)
    }
}

/// PMAT-216: Implement TracedForward trait for CPU backend
impl TracedForward for AprTransformer {
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
        // Delegate to the immutable method (CPU doesn't need mutation)
        AprTransformer::forward_traced(self, tokens)
    }
}

// Tests shattered to tests/ directory (PMAT-803)
#[cfg(test)]
mod tests;
