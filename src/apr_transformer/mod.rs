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
mod dequant;
mod helpers;
mod loader;
mod q4_simd;
pub use config::{
    AprKVCache, AprTransformerConfig, AprTransformerLayer, GenerateConfig, Q4KLayerWeights,
};
use dequant::{dequantize_q4_k_apr, dequantize_q6_k_apr};
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
            .map(|s| s.to_lowercase())
            .filter(|s| s != "auto" && !s.is_empty())  // "Auto" is not a valid architecture
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

                match dtype {
                    // Q4_K: converter dtype=8 or APR v2 native dtype=12
                    8 | 12 => {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q4_k_apr(tensor_data, num_elements)
                    },
                    // Q6_K: converter dtype=9 or APR v2 native dtype=14
                    9 | 14 => {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q6_k_apr(tensor_data, num_elements)
                    },
                    // F32 (dtype=0) or other: interpret as raw F32
                    _ => tensor_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                }
            })
        };

        // PMAT-103 FIX: Helper to get raw Q4K bytes (no dequantization) for fused kernel
        // Returns None if tensor is not Q4K/Q5K/Q6K format
        // APR dtype mapping (from GgufToAprQ4KConverter):
        //   8 = Q4_K (GGML 12) or Q5_K (GGML 13)
        //   9 = Q6_K (GGML 14)
        //  10 = Q8_0 (GGML 8)
        //  12 = Q4_K (APR v2 native)
        let get_q4k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q4K tensors from either converter dtype (8) or APR v2 native dtype (12)
                if *dtype != 8 && *dtype != 12 {
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
        let get_q6k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q6K tensors from either converter dtype (9) or APR v2 native dtype (14)
                if *dtype != 9 && *dtype != 14 {
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
                    eprintln!("  {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
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

        // PMAT-086: Debug - check which embedding tensor names exist
        let embed_names = [
            "model.embed_tokens.weight",
            "token_embd.weight",
            "tok_embeddings.weight",
        ];
        if debug_enabled {
            for name in &embed_names {
                if let Some((offset, size, dims, dtype)) = tensors.get(*name) {
                    eprintln!("[DEBUG] Found embedding {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
                }
            }
        }

        // Try to load token embedding
        let token_embedding_raw = get_f32_tensor("model.embed_tokens.weight")
            .or_else(|| get_f32_tensor("token_embd.weight"))
            .or_else(|| get_f32_tensor("tok_embeddings.weight"))
            .unwrap_or_else(|| {
                if debug_enabled {
                    eprintln!("[DEBUG] WARNING: No embedding tensor found! Using zeros.");
                }
                vec![0.0; vocab_size * hidden_dim]
            });

        // PMAT-086 FIX: APR stores GGUF data in row-major [vocab_size, hidden_dim] layout
        // even though the dims metadata says [hidden_dim, vocab_size] (GGML column-major convention)
        // The data is already correct - DO NOT transpose!
        let token_embedding = token_embedding_raw;

        if debug_enabled {
            eprintln!(
                "[DEBUG] token_embedding loaded: {} elements, first 5: {:?}",
                token_embedding.len(),
                &token_embedding[..5.min(token_embedding.len())]
            );
        }

        // Load output norm
        let output_norm_weight = get_f32_tensor("model.norm.weight")
            .or_else(|| get_f32_tensor("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden_dim]);

        // Debug: check output.weight / lm_head.weight
        if debug_enabled {
            for name in &["output.weight", "lm_head.weight"] {
                if let Some((offset, size, dims, dtype)) = tensors.get(*name) {
                    eprintln!("[DEBUG] Found lm_head {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
                }
            }
        }

        // Load LM head
        // For tied embeddings (common in Qwen, LLaMA models), use embed_tokens as fallback
        let lm_head_raw = get_f32_tensor("lm_head.weight")
            .or_else(|| get_f32_tensor("output.weight"));
        let (lm_head_raw, used_tied_weights) = if let Some(lm_head) = lm_head_raw {
            (lm_head, false)
        } else {
            // Weight tying: use embedding weights for lm_head
            let tied = get_f32_tensor("model.embed_tokens.weight")
                .or_else(|| get_f32_tensor("token_embd.weight"));
            if let Some(t) = tied {
                (t, true)
            } else {
                (vec![0.0; hidden_dim * vocab_size], false)
            }
        };
        if debug_enabled {
            if used_tied_weights {
                eprintln!("[DEBUG] Using tied weights: embedding -> lm_head");
            }
            if lm_head_raw.iter().all(|&x| x == 0.0) {
                eprintln!("[DEBUG] WARNING: No lm_head tensor found! Using zeros.");
            }
            eprintln!(
                "[DEBUG] lm_head_raw: {} elements, first 5: {:?}",
                lm_head_raw.len(),
                &lm_head_raw[..5.min(lm_head_raw.len())]
            );
        }
        // PMAT-086 FIX: APR stores GGUF data in row-major [vocab_size, hidden_dim] layout
        // even though the dims metadata says [hidden_dim, vocab_size] (GGML column-major convention)
        // The data is already correct - DO NOT transpose!
        let lm_head_weight = lm_head_raw;

        // PMAT-103: Load lm_head Q4K/Q6K raw bytes for fused kernel inference
        let lm_head_weight_q4k =
            get_q4k_raw_bytes("lm_head.weight").or_else(|| get_q4k_raw_bytes("output.weight"));
        let lm_head_weight_q6k =
            get_q6k_raw_bytes("lm_head.weight").or_else(|| get_q6k_raw_bytes("output.weight"));
        if debug_enabled {
            if lm_head_weight_q4k.is_some() {
                eprintln!("[DEBUG] Loaded lm_head Q4K raw bytes for fused kernel");
            } else if lm_head_weight_q6k.is_some() {
                eprintln!("[DEBUG] Loaded lm_head Q6K raw bytes for fused kernel");
            }
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
            let qkv_bias = if let Some(fused_bias) = get_f32_tensor(&format!("{hf_prefix}.self_attn.qkv_proj.bias")) {
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
            // Now includes separate Q/K/V weights for fused QKV projection
            let q4k_attn_q = get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_q.weight"));
            let q4k_attn_k = get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_k.weight"));
            let q4k_attn_v = get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_v.weight"));
            let q6k_attn_v = get_q6k_raw_bytes(&format!("{gguf_prefix}.attn_v.weight"));
            let q4k_attn_output = get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_output.weight"));
            let q4k_ffn_gate = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_gate.weight"));
            let q4k_ffn_up = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight"));
            let q4k_ffn_down = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight"));
            // Q6K fallback for tensors that aren't Q4K (common in mixed quantization models)
            let q6k_ffn_down = get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight"));
            let q6k_ffn_up = get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight"));

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
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // Out of vocab - return zeros
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// RMSNorm (Root Mean Square Layer Normalization)
    ///
    /// Used by Qwen2, LLaMA, Mistral, and most modern LLMs.
    /// Formula: output = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// PMAT-094: Fixed five-whys root cause - was using LayerNorm (mean subtraction)
    /// instead of RMSNorm which caused garbage output for Qwen2 models.
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for s in 0..seq_len {
            let start = s * hidden_dim;
            let slice = &input[start..start + hidden_dim];

            // RMSNorm: compute root mean square (no mean subtraction!)
            let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

            // Normalize and scale
            for (i, &x) in slice.iter().enumerate() {
                let normalized = x / rms;
                let scaled = normalized * weight[i];
                let shifted = if let Some(b) = bias {
                    scaled + b[i]
                } else {
                    scaled
                };
                output.push(shifted);
            }
        }

        output
    }

    /// Matrix multiplication: output[out_dim] = weight[out_dim, in_dim] @ input[in_dim]
    ///
    /// PMAT-095 FIX: Weights are now stored in matvec-optimal [out_dim, in_dim] format.
    ///
    /// PMAT-103 FIX: Zero-copy implementation using raw slice operations.
    /// Previous implementations had O(n) allocation overhead per matmul call.
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let expected_size = in_dim * out_dim;

        if weight.len() != expected_size {
            return self.matmul_scalar(input, weight, in_dim, out_dim);
        }

        let mut output = vec![0.0f32; seq_len * out_dim];

        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];
            let out_start = s * out_dim;

            // PMAT-103: Unrolled dot product for better cache utilization
            // Process 4 output elements at a time when possible
            let out_chunks = out_dim / 4;
            let out_remainder = out_dim % 4;

            for o_chunk in 0..out_chunks {
                let o_base = o_chunk * 4;
                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut sum3 = 0.0f32;

                let w0_start = (o_base) * in_dim;
                let w1_start = (o_base + 1) * in_dim;
                let w2_start = (o_base + 2) * in_dim;
                let w3_start = (o_base + 3) * in_dim;

                for i in 0..in_dim {
                    let x = input_slice[i];
                    sum0 += x * weight[w0_start + i];
                    sum1 += x * weight[w1_start + i];
                    sum2 += x * weight[w2_start + i];
                    sum3 += x * weight[w3_start + i];
                }

                output[out_start + o_base] = sum0;
                output[out_start + o_base + 1] = sum1;
                output[out_start + o_base + 2] = sum2;
                output[out_start + o_base + 3] = sum3;
            }

            // Handle remainder
            for o in (out_dim - out_remainder)..out_dim {
                let w_start = o * in_dim;
                let mut sum = 0.0f32;
                for i in 0..in_dim {
                    sum += input_slice[i] * weight[w_start + i];
                }
                output[out_start + o] = sum;
            }
        }

        output
    }

    /// Scalar fallback for matmul (used when trueno fails)
    ///
    /// PMAT-095: Weight is [out_dim, in_dim] row-major format
    #[allow(clippy::unused_self)]
    fn matmul_scalar(
        &self,
        input: &[f32],
        weight: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let mut output = Vec::with_capacity(seq_len * out_dim);

        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];

            for o in 0..out_dim {
                let mut sum = 0.0;
                for (i, &input_val) in input_slice.iter().enumerate() {
                    // PMAT-095: Weight is [out_dim, in_dim] row-major
                    let weight_idx = o * in_dim + i;
                    if weight_idx < weight.len() {
                        sum += input_val * weight[weight_idx];
                    }
                }
                output.push(sum);
            }
        }

        output
    }

    /// Add bias in-place
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        let dim = bias.len();
        for (i, val) in data.iter_mut().enumerate() {
            *val += bias[i % dim];
        }
    }

    /// GELU activation (tanh approximation)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;

        for x in data.iter_mut() {
            let x3 = *x * *x * *x;
            let inner = SQRT_2_OVER_PI * (*x + GELU_COEFF * x3);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Apply Rotary Position Embedding (RoPE) to Q or K vectors
    ///
    /// RoPE encodes position information by rotating pairs of elements
    /// with position-dependent angles.
    fn apply_rope_f32(&self, x: &mut [f32], position: usize, num_heads: usize, head_dim: usize) {
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        for h in 0..num_heads {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            if idx2_start + half_dim > x.len() {
                continue;
            }

            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
            }
        }
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
                        matmul_q4k_rowmajor(q4k_bytes, input_slice, hidden_dim, hidden_dim);
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
            let ffn_output = if let Some(ref _gate_weight) = layer.ffn_gate_weight {
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
                            );
                            output.extend(pos_out);
                        }
                        output
                    } else {
                        self.matmul(
                            &ffn_input,
                            layer.ffn_gate_weight.as_ref().expect("gate weight"),
                            hidden_dim,
                            intermediate_dim,
                        )
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
                        );
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
                        );
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
                        );
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
                            );
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
                            );
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

            // 2b. QKV projection (single token)
            // PMAT-103: Use fused Q4K kernels for separate Q, K, V weights when available
            let kv_size = num_kv_heads * head_dim;
            let (mut q, mut k, v) = if let Some(q4k) = q4k_layer {
                // Try Q4K fused kernels for Q, K
                let q = if let Some(ref q_bytes) = q4k.attn_q_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: Q projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q_bytes, &normed, hidden_dim, hidden_dim)
                } else {
                    // Fallback to F32 for Q (should not happen for GGUF models)
                    let q_weight = &layer.qkv_weight[0..hidden_dim * hidden_dim];
                    self.matmul(&normed, q_weight, hidden_dim, hidden_dim)
                };

                let k = if let Some(ref k_bytes) = q4k.attn_k_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: K projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(k_bytes, &normed, kv_size, hidden_dim)
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
                    matmul_q4k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)
                } else if let Some(ref v_bytes) = q4k.attn_v_weight_q6k {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: V projection using Q6K fused kernel");
                    }
                    matmul_q6k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)
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

            // PMAT-103: Apply RoPE to Q and K at current position
            // This was missing, causing garbage output
            self.apply_rope_f32(&mut q, position, num_heads, head_dim);
            self.apply_rope_f32(&mut k, position, num_kv_heads, head_dim);

            // 2c. Append K, V to cache (K now has RoPE applied)
            cache.append(layer_idx, &k, &v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let seq_len = cache.len();

            // Simplified attention: compute Q·K^T / sqrt(d), softmax, then V
            let mut attn_out = vec![0.0f32; hidden_dim];

            // PMAT-103: SIMD-accelerated attention computation
            let scale = 1.0 / (head_dim as f32).sqrt();

            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads; // GQA mapping
                let q_start = h * head_dim;
                let q_slice = &q[q_start..q_start + head_dim]; // q is now Vec with RoPE applied

                // Compute attention scores with SIMD dot product
                let mut scores = Vec::with_capacity(seq_len);
                for pos in 0..seq_len {
                    let k_start = pos * kv_size + kv_head * head_dim;
                    let k_slice = &k_cache[k_start..k_start + head_dim];
                    // SIMD dot product (AVX2 when available)
                    let dot = simd_dot_f32(q_slice, k_slice);
                    scores.push(dot * scale);
                }

                // Causal mask: only attend to positions <= current
                for pos in (position + 1)..seq_len {
                    scores[pos] = f32::NEG_INFINITY;
                }

                // Softmax (scalar - typically small seq_len during decode)
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

                // Weighted sum of V with SIMD accumulation
                let attn_out_head = &mut attn_out[q_start..q_start + head_dim];
                for pos in 0..seq_len {
                    let v_start = pos * kv_size + kv_head * head_dim;
                    let v_slice = &v_cache[v_start..v_start + head_dim];
                    // SIMD weighted accumulation (AVX2 when available)
                    simd_add_weighted(attn_out_head, v_slice, exp_scores[pos]);
                }
            }

            // 2e. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available (single token path)
            let mut attn_output = if !force_f32 {
                if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.attn_output_weight.as_ref()) {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q4k_bytes, &attn_out, hidden_dim, hidden_dim)
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

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
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
            let ffn_output = if let Some(ref _gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // PMAT-103: Check for Q4K gate weight
                let gate = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                    } else {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 fallback (slow!)");
                        }
                        self.matmul(
                            &ffn_input,
                            layer.ffn_gate_weight.as_ref().expect("gate weight"),
                            hidden_dim,
                            intermediate_dim,
                        )
                    }
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(
                        &ffn_input,
                        layer.ffn_gate_weight.as_ref().expect("gate weight"),
                        hidden_dim,
                        intermediate_dim,
                    )
                };

                // PMAT-103: Check for Q4K/Q6K up weight
                let up = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                    } else if let Some(q6k_bytes) =
                        q4k_layer.and_then(|q| q.ffn_up_weight_q6k.as_ref())
                    {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q6K fused kernel");
                        }
                        matmul_q6k_rowmajor(q6k_bytes, &ffn_input, intermediate_dim, hidden_dim)
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
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
                    } else if let Some(q6k_bytes) =
                        q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref())
                    {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q6K fused kernel");
                        }
                        matmul_q6k_rowmajor(q6k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
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
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
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
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
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
        }
        if trace_enabled {
            eprintln!(
                "[TRACE-CACHE] pos={}: {} layers took {:?}",
                position,
                self.layers.len(),
                layers_start.elapsed()
            );
        }

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
                matmul_q4k_rowmajor(q4k_bytes, &normed, self.config.vocab_size, hidden_dim)
            } else if let Some(ref q6k_bytes) = self.lm_head_weight_q6k {
                let result =
                    matmul_q6k_rowmajor(q6k_bytes, &normed, self.config.vocab_size, hidden_dim);
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

    /// Generate tokens using KV cache for efficiency (Y4)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate_with_cache(&self, prompt: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut cache = AprKVCache::new(&self.config);
        let mut output = prompt.to_vec();

        // PMAT-103 FIX: Process prompt tokens and KEEP the logits from the last one.
        // Previously we threw away all logits (`let _ = ...`) and then reprocessed
        // the last prompt token at the same position, corrupting the KV cache.
        let mut logits = Vec::new();

        // PMAT-103 TRACE: Measure per-token timing to verify O(n) vs O(n²)
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();
        if trace_enabled {
            eprintln!("[TRACE] Processing {} prompt tokens...", prompt.len());
        }

        for (pos, &token) in prompt.iter().enumerate() {
            let start = std::time::Instant::now();
            logits = self.forward_with_cache(token, &mut cache, pos)?;
            if trace_enabled {
                eprintln!("[TRACE] Prompt token {}: {:?}", pos, start.elapsed());
            }
        }

        // Generate new tokens using the logits we already have
        for i in 0..config.max_tokens {
            // Sample from current logits (which predict the NEXT token)
            let next_token = if config.temperature == 0.0 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = scaled.iter().map(|s| (s - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            };

            output.push(next_token);

            // Check for EOS tokens
            if next_token == 0 || next_token == 2 || next_token == 151645 || next_token == 151643 {
                break;
            }

            // If we need more tokens, process this one to get logits for the next
            if i < config.max_tokens - 1 {
                // Position is output.len() - 1 = prompt.len() + (i + 1) - 1 = prompt.len() + i
                let start = std::time::Instant::now();
                logits = self.forward_with_cache(next_token, &mut cache, output.len() - 1)?;
                if trace_enabled {
                    eprintln!(
                        "[TRACE] Gen token {} (pos {}): {:?}",
                        i,
                        output.len() - 1,
                        start.elapsed()
                    );
                }
            }
        }

        if trace_enabled {
            eprintln!(
                "[TRACE] Generation complete. Total output tokens: {}",
                output.len()
            );
        }

        Ok(output)
    }
}

/// Convert from `GGUFTransformer` to APR format
///
/// This dequantizes all GGUF weights to F32 for WASM compatibility.
#[cfg(feature = "default")]
impl From<&crate::gguf::GGUFTransformer> for AprTransformer {
    fn from(gguf: &crate::gguf::GGUFTransformer) -> Self {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
            })
            .collect();

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }
}
// Tests shattered to tests/ directory (PMAT-803)
#[cfg(test)]
mod tests;
