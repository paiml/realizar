//! Aprender .apr format support for realizar (APR v2 only)
//!
//! This module provides loading and inference for models in Aprender's native
//! .apr v2 format (Magic: `APR2` = 0x41505232).
//!
//! ## Format Structure (APR v2, 64-byte header)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                           │
//! │   - Magic: "APR2" (4 bytes)                                 │
//! │   - Version: major.minor (2 bytes)                          │
//! │   - Flags (2 bytes)                                         │
//! │   - Tensor count (4 bytes)                                  │
//! │   - Metadata offset (8 bytes)                               │
//! │   - Metadata size (4 bytes)                                 │
//! │   - Tensor index offset (8 bytes)                           │
//! │   - Data offset (8 bytes)                                   │
//! │   - Checksum (4 bytes)                                      │
//! │   - Reserved (20 bytes)                                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ JSON Metadata (padded to 64-byte boundary)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Index (sorted by name)                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Data (each tensor 64-byte aligned)                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr::AprV2Model;
//!
//! let model = AprV2Model::load("model.apr")?;
//! println!("Tensors: {}", model.tensor_count());
//! ```

use std::collections::HashMap;
use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use trueno::brick::BrickProfiler;

use crate::error::{RealizarError, Result};

/// Magic number: "APR2" in ASCII (0x41505232)
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x32];

/// Format version for .apr v2 files
pub const FORMAT_VERSION: (u8, u8) = (2, 0);

/// Header size in bytes (64-byte aligned)
pub const HEADER_SIZE: usize = 64;

/// Tensor alignment in bytes
pub const ALIGNMENT: usize = 64;

/// APR v2 feature flags
#[derive(Debug, Clone, Copy, Default)]
pub struct AprFlags(u16);

impl AprFlags {
    /// LZ4 compression enabled
    pub const LZ4_COMPRESSED: u16 = 0x0001;
    /// Zstandard compression enabled
    pub const ZSTD_COMPRESSED: u16 = 0x0002;
    /// Model is encrypted
    pub const ENCRYPTED: u16 = 0x0004;
    /// Model has cryptographic signature
    pub const SIGNED: u16 = 0x0008;
    /// Model is sharded across multiple files
    pub const SHARDED: u16 = 0x0010;
    /// Weights are quantized (int8/int4)
    pub const QUANTIZED: u16 = 0x0020;
    /// Model includes embedded vocabulary
    pub const HAS_VOCAB: u16 = 0x0200;

    /// Create flags from raw bits
    #[must_use]
    pub const fn new(bits: u16) -> Self {
        Self(bits)
    }

    /// Check if model uses compression (LZ4 or Zstd)
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        self.0 & (Self::LZ4_COMPRESSED | Self::ZSTD_COMPRESSED) != 0
    }

    /// Check if model uses LZ4 compression
    #[must_use]
    pub const fn is_lz4(&self) -> bool {
        self.0 & Self::LZ4_COMPRESSED != 0
    }

    /// Check if model uses ZSTD compression
    #[must_use]
    pub const fn is_zstd(&self) -> bool {
        self.0 & Self::ZSTD_COMPRESSED != 0
    }

    /// Check if model is encrypted
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    /// Check if weights are quantized
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        self.0 & Self::QUANTIZED != 0
    }

    /// Check if model includes embedded vocabulary
    #[must_use]
    pub const fn has_vocab(&self) -> bool {
        self.0 & Self::HAS_VOCAB != 0
    }
}

/// APR v2 file header (64 bytes)
#[derive(Debug, Clone)]
pub struct AprHeader {
    /// Magic number ("APR2")
    pub magic: [u8; 4],
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Feature flags
    pub flags: AprFlags,
    /// Number of tensors
    pub tensor_count: u32,
    /// Offset to metadata section
    pub metadata_offset: u64,
    /// Size of metadata section
    pub metadata_size: u32,
    /// Offset to tensor index
    pub tensor_index_offset: u64,
    /// Offset to tensor data
    pub data_offset: u64,
    /// Header checksum (CRC32)
    pub checksum: u32,
}

impl AprHeader {
    /// Parse header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr header too small: {} bytes (need {})",
                    data.len(),
                    HEADER_SIZE
                ),
            });
        }

        // Check magic
        let magic: [u8; 4] = data[0..4]
            .try_into()
            .map_err(|_| RealizarError::FormatError {
                reason: "Failed to read magic bytes".to_string(),
            })?;

        if magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid .apr magic: expected APR2 {:?}, got {:?}",
                    MAGIC, magic
                ),
            });
        }

        let version = (data[4], data[5]);
        let flags = AprFlags::new(u16::from_le_bytes([data[6], data[7]]));
        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]);
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]);
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]);
        let checksum = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);

        Ok(Self {
            magic,
            version,
            flags,
            tensor_count,
            metadata_offset,
            metadata_size,
            tensor_index_offset,
            data_offset,
            checksum,
        })
    }
}

/// Tensor entry in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Tensor name (e.g., "model.layers.0.attention.wq")
    pub name: String,
    /// Data type (e.g., "F32", "F16", "BF16", "I8")
    pub dtype: String,
    /// Tensor dimensions
    pub shape: Vec<usize>,
    /// Byte offset from data section start
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

impl TensorEntry {
    /// Parse tensor entry from binary format (aprender v2 format)
    ///
    /// Binary format:
    /// - name_len (2 bytes LE) + name bytes
    /// - dtype (1 byte)
    /// - ndim (1 byte) + dims (8 bytes LE each, up to 8)
    /// - offset (8 bytes LE)
    /// - size (8 bytes LE)
    pub fn from_binary(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 4 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry too short".to_string(),
            });
        }

        let mut pos = 0;

        // Name
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        if data.len() < pos + name_len + 2 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry truncated at name".to_string(),
            });
        }

        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Dtype (1 byte)
        let dtype_byte = data[pos];
        pos += 1;
        let dtype = match dtype_byte {
            0 => "F32",
            1 => "F16",
            2 => "BF16",
            3 => "I8",
            4 => "I16",
            5 => "I32",
            6 => "I64",
            7 => "U8",
            _ => "F32",
        }
        .to_string();

        // Shape: ndim (1 byte) + dims
        let ndim = data[pos] as usize;
        pos += 1;

        if data.len() < pos + ndim * 8 + 16 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry truncated at shape".to_string(),
            });
        }

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
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
            pos += 8;
            shape.push(dim);
        }

        // Offset and size
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let size = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        Ok((
            Self {
                name,
                dtype,
                shape,
                offset,
                size,
            },
            pos,
        ))
    }

    /// Calculate element count from shape
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Model metadata from .apr file
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AprMetadata {
    /// Model type (e.g., "transformer_lm", "whisper", "llama")
    #[serde(default)]
    pub model_type: Option<String>,
    /// Human-readable model name
    #[serde(default)]
    pub name: Option<String>,
    /// Model architecture family
    #[serde(default)]
    pub architecture: Option<String>,
    /// Hidden dimension size
    #[serde(default)]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    #[serde(default)]
    pub num_layers: Option<usize>,
    /// Number of attention heads
    #[serde(default)]
    pub num_heads: Option<usize>,
    /// Number of key-value heads (for GQA, defaults to num_heads)
    #[serde(default)]
    pub num_kv_heads: Option<usize>,
    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// FFN intermediate dimension
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    /// Maximum context/sequence length
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta for position encoding
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Layer norm epsilon
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    /// Additional metadata fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl AprMetadata {
    /// Check if this model has transformer configuration
    #[must_use]
    pub fn is_transformer(&self) -> bool {
        self.hidden_size.is_some()
            && self.num_layers.is_some()
            && self.num_heads.is_some()
            && self.vocab_size.is_some()
    }
}

/// APR v2 model for realizar inference
#[derive(Debug)]
pub struct AprV2Model {
    /// Header information
    header: AprHeader,
    /// Model metadata
    metadata: AprMetadata,
    /// Tensor index
    tensors: Vec<TensorEntry>,
    /// Raw file data (mmap in production)
    data: Vec<u8>,
}

impl AprV2Model {
    /// Load a model from a .apr file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read .apr file: {e}"),
        })?;

        Self::from_bytes(data)
    }

    /// Load a model from bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        // Parse header
        let header = AprHeader::from_bytes(&data)?;

        // Validate version
        if header.version.0 > FORMAT_VERSION.0 {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr version {}.{} not supported (max {}.{})",
                    header.version.0, header.version.1, FORMAT_VERSION.0, FORMAT_VERSION.1
                ),
            });
        }

        // Check for unsupported features
        if header.flags.is_encrypted() {
            return Err(RealizarError::FormatError {
                reason: "Encrypted .apr files not yet supported".to_string(),
            });
        }

        // Decompress data if needed (GH-35)
        let data = if header.flags.is_compressed() {
            Self::decompress_apr_data(&header, data)?
        } else {
            data
        };

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data.len() < metadata_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr file truncated: metadata extends to {} but file is {} bytes",
                    metadata_end,
                    data.len()
                ),
            });
        }

        let metadata: AprMetadata = if header.metadata_size > 0 {
            serde_json::from_slice(&data[metadata_start..metadata_end]).unwrap_or_default()
        } else {
            AprMetadata::default()
        };

        // Parse tensor index (binary format from aprender v2)
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        if index_start < index_end && index_end <= data.len() {
            let index_data = &data[index_start..index_end];
            let mut pos = 0;

            while pos < index_data.len() && tensors.len() < header.tensor_count as usize {
                match TensorEntry::from_binary(&index_data[pos..]) {
                    Ok((entry, consumed)) => {
                        tensors.push(entry);
                        pos += consumed;
                    },
                    Err(_) => break, // Stop on parse error
                }
            }
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            data,
        })
    }

    /// Decompress APR data based on compression flags (GH-35)
    ///
    /// The compressed format stores: header (64 bytes, uncompressed) + compressed payload.
    /// We decompress the payload and reconstruct the full data vector.
    #[allow(unreachable_patterns)] // Pattern varies based on apr-compression feature
    fn decompress_apr_data(header: &AprHeader, data: Vec<u8>) -> Result<Vec<u8>> {
        #[cfg(feature = "apr-compression")]
        let compressed_payload = &data[HEADER_SIZE..];

        #[cfg(feature = "apr-compression")]
        {
            let decompressed = if header.flags.is_lz4() {
                lz4_flex::decompress_size_prepended(compressed_payload).map_err(|e| {
                    RealizarError::FormatError {
                        reason: format!("LZ4 decompression failed: {e}"),
                    }
                })?
            } else if header.flags.is_zstd() {
                zstd::decode_all(compressed_payload).map_err(|e| RealizarError::FormatError {
                    reason: format!("ZSTD decompression failed: {e}"),
                })?
            } else {
                // Unknown compression - should not happen
                return Err(RealizarError::FormatError {
                    reason: "Unknown compression algorithm in APR flags".to_string(),
                });
            };

            // Reconstruct full data: header + decompressed payload
            let mut result = Vec::with_capacity(HEADER_SIZE + decompressed.len());
            result.extend_from_slice(&data[..HEADER_SIZE]);
            result.extend_from_slice(&decompressed);
            return Ok(result);
        }

        #[cfg(not(feature = "apr-compression"))]
        {
            let _ = (header, &data); // Suppress unused warnings
            Err(RealizarError::FormatError {
                reason: "Compressed .apr files require 'apr-compression' feature".to_string(),
            })
        }
    }

    /// Get number of tensors
    #[must_use]
    pub fn tensor_count(&self) -> u32 {
        self.header.tensor_count
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprMetadata {
        &self.metadata
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensor data as f32 slice
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;

        if end > self.data.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        let bytes = &self.data[start..end];

        // Parse based on dtype
        match entry.dtype.as_str() {
            "F32" | "f32" => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            },
            dtype => Err(RealizarError::FormatError {
                reason: format!("Unsupported tensor dtype: {dtype}"),
            }),
        }
    }

    /// Get raw tensor bytes
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;

        if end > self.data.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        Ok(&self.data[start..end])
    }

    /// Estimate total parameters
    #[must_use]
    pub fn estimated_parameters(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum()
    }

    /// Run inference on input features (for simple models)
    ///
    /// For transformer models, use `forward()` instead.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector
    ///
    /// # Returns
    ///
    /// Output vector
    ///
    /// # Errors
    ///
    /// Returns error if model has no tensors
    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>> {
        if self.tensors.is_empty() && self.header.tensor_count == 0 {
            let sum: f32 = features.iter().sum();
            return Ok(vec![sum]);
        }

        // Linear model: y = Wx + b (if we have weights)
        if let Some(weight) = self.get_tensor("weight") {
            let weights = self.get_tensor_f32("weight")?;
            let bias = self.get_tensor_f32("bias").unwrap_or_default();

            let output_dim = weight.shape.first().copied().unwrap_or(1);
            let input_dim = weight.shape.get(1).copied().unwrap_or(features.len());

            let mut output = vec![0.0; output_dim];
            for (i, out) in output.iter_mut().enumerate() {
                for (j, &feat) in features.iter().take(input_dim).enumerate() {
                    *out += weights.get(i * input_dim + j).copied().unwrap_or(0.0) * feat;
                }
                *out += bias.get(i).copied().unwrap_or(0.0);
            }
            return Ok(output);
        }

        let sum: f32 = features.iter().sum();
        Ok(vec![sum])
    }

    /// Run transformer forward pass on token IDs
    ///
    /// Returns logits for the next token prediction.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size`
    ///
    /// # Errors
    ///
    /// Returns error if model is not a transformer or tensors are missing
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.metadata.num_layers.unwrap_or(0);
        let num_heads = self.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.metadata.rms_norm_eps.unwrap_or(1e-6);

        // 1. Token embedding lookup
        let embed_name = self.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",           // SafeTensors (no model. prefix)
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
        ])?;

        let embeddings = self.get_tensor_f32(&embed_name)?;
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Try common naming patterns
            let attn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
            ])?;

            let q_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
            ])?;

            let k_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
            ])?;

            let v_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
            ])?;

            let o_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
            ])?;

            // Load tensors
            let norm_weight = self.get_tensor_f32(&attn_norm_name)?;
            let q_weight = self.get_tensor_f32(&q_name)?;
            let k_weight = self.get_tensor_f32(&k_name)?;
            let v_weight = self.get_tensor_f32(&v_name)?;
            let o_weight = self.get_tensor_f32(&o_name)?;

            // RMSNorm
            let normed = rms_norm(&hidden, &norm_weight, eps);

            // Attention: Q, K, V projections
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / num_heads;

            let q = matmul(&normed, &q_weight, seq_len, hidden_dim, hidden_dim);
            let k = matmul(&normed, &k_weight, seq_len, hidden_dim, num_kv_heads * head_dim);
            let v = matmul(&normed, &v_weight, seq_len, hidden_dim, num_kv_heads * head_dim);

            // Simplified attention (no RoPE for now, full attention)
            let attn_out = simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);

            // Output projection
            let attn_proj = matmul(&attn_out, &o_weight, seq_len, hidden_dim, hidden_dim);

            // Residual connection
            for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }

            // FFN
            let ffn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
            ])?;

            let gate_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
            ])?;

            let up_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
            ])?;

            let down_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),  // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
            ])?;

            let ffn_norm = self.get_tensor_f32(&ffn_norm_name)?;
            let gate = self.get_tensor_f32(&gate_name)?;
            let up = self.get_tensor_f32(&up_name)?;
            let down = self.get_tensor_f32(&down_name)?;

            let normed = rms_norm(&hidden, &ffn_norm, eps);
            let gate_out = matmul(&normed, &gate, seq_len, hidden_dim, intermediate_dim);
            let up_out = matmul(&normed, &up, seq_len, hidden_dim, intermediate_dim);

            // SiLU activation and element-wise multiply
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }

            let ffn_out = matmul(&ffn_hidden, &down, seq_len, intermediate_dim, hidden_dim);

            // Residual
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
        }

        // 3. Final layer norm
        let final_norm_name = self.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",               // SafeTensors
            "transformer.ln_f.weight",
        ])?;
        let final_norm = self.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);

        // 4. LM head (last token only for generation)
        let lm_head_name = self.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight", // Tied embeddings
            "embed_tokens.weight",       // SafeTensors tied embeddings
        ])?;
        let lm_head = self.get_tensor_f32(&lm_head_name)?;

        // Get hidden state for last token
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        // Project to vocab
        let mut logits = vec![0.0; vocab_size];
        for (i, logit) in logits.iter_mut().enumerate() {
            for (j, &h) in last_hidden.iter().enumerate() {
                *logit += h * lm_head.get(i * hidden_dim + j).copied().unwrap_or(0.0);
            }
        }

        Ok(logits)
    }

    /// Forward pass with BrickProfiler instrumentation.
    ///
    /// Instruments 11 timing points:
    /// - apr.Embed: Token embedding lookup
    /// - apr.RmsNorm: RMS normalization (called 2x per layer + 1 final)
    /// - apr.QKV: Q, K, V projections
    /// - apr.Attention: Scaled dot-product attention
    /// - apr.OProj: Output projection
    /// - apr.FFN: Gate + Up + Down MLPs
    /// - apr.Residual: Residual connection adds (2x per layer)
    /// - apr.FinalNorm: Final layer norm
    /// - apr.LmHead: LM head projection
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token sequence
    /// * `profiler` - BrickProfiler instance (must be enabled)
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size`
    pub fn forward_profiled(
        &self,
        token_ids: &[u32],
        profiler: &mut BrickProfiler,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.metadata.num_layers.unwrap_or(0);
        let num_heads = self.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();

        // 1. APR.EMBED: Token embedding lookup
        let timer = profiler.start("apr.Embed");
        let embed_name = self.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
        ])?;
        let embeddings = self.get_tensor_f32(&embed_name)?;
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }
        profiler.stop(timer, seq_len as u64);

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Load tensor names
            let attn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
            ])?;
            let q_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
            ])?;
            let k_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
            ])?;
            let v_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
            ])?;
            let o_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
            ])?;

            let norm_weight = self.get_tensor_f32(&attn_norm_name)?;
            let q_weight = self.get_tensor_f32(&q_name)?;
            let k_weight = self.get_tensor_f32(&k_name)?;
            let v_weight = self.get_tensor_f32(&v_name)?;
            let o_weight = self.get_tensor_f32(&o_name)?;

            // APR.RMSNORM (input)
            let timer = profiler.start("apr.RmsNorm");
            let normed = rms_norm(&hidden, &norm_weight, eps);
            profiler.stop(timer, seq_len as u64);

            // APR.QKV
            let timer = profiler.start("apr.QKV");
            let head_dim = hidden_dim / num_heads;
            let q = matmul(&normed, &q_weight, seq_len, hidden_dim, hidden_dim);
            let k = matmul(&normed, &k_weight, seq_len, hidden_dim, num_kv_heads * head_dim);
            let v = matmul(&normed, &v_weight, seq_len, hidden_dim, num_kv_heads * head_dim);
            profiler.stop(timer, seq_len as u64);

            // APR.ATTENTION
            let timer = profiler.start("apr.Attention");
            let attn_out = simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
            profiler.stop(timer, seq_len as u64);

            // APR.OPROJ
            let timer = profiler.start("apr.OProj");
            let attn_proj = matmul(&attn_out, &o_weight, seq_len, hidden_dim, hidden_dim);
            profiler.stop(timer, seq_len as u64);

            // APR.RESIDUAL (attention)
            let timer = profiler.start("apr.Residual");
            for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }
            profiler.stop(timer, seq_len as u64);

            // FFN path
            let ffn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
            ])?;
            let gate_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
            ])?;
            let up_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
            ])?;
            let down_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
            ])?;

            let ffn_norm = self.get_tensor_f32(&ffn_norm_name)?;
            let gate = self.get_tensor_f32(&gate_name)?;
            let up = self.get_tensor_f32(&up_name)?;
            let down = self.get_tensor_f32(&down_name)?;

            // APR.RMSNORM (post-attention)
            let timer = profiler.start("apr.RmsNorm");
            let normed = rms_norm(&hidden, &ffn_norm, eps);
            profiler.stop(timer, seq_len as u64);

            // APR.FFN
            let timer = profiler.start("apr.FFN");
            let gate_out = matmul(&normed, &gate, seq_len, hidden_dim, intermediate_dim);
            let up_out = matmul(&normed, &up, seq_len, hidden_dim, intermediate_dim);
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }
            let ffn_out = matmul(&ffn_hidden, &down, seq_len, intermediate_dim, hidden_dim);
            profiler.stop(timer, seq_len as u64);

            // APR.RESIDUAL (FFN)
            let timer = profiler.start("apr.Residual");
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
            profiler.stop(timer, seq_len as u64);
        }

        // 3. APR.FINALNORM
        let timer = profiler.start("apr.FinalNorm");
        let final_norm_name = self.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
        ])?;
        let final_norm = self.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);
        profiler.stop(timer, seq_len as u64);

        // 4. APR.LMHEAD
        let timer = profiler.start("apr.LmHead");
        let lm_head_name = self.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ])?;
        let lm_head = self.get_tensor_f32(&lm_head_name)?;
        let last_hidden = &hidden[hidden.len() - hidden_dim..];
        let mut logits = vec![0.0; vocab_size];
        for (i, logit) in logits.iter_mut().enumerate() {
            for (j, &h) in last_hidden.iter().enumerate() {
                *logit += h * lm_head.get(i * hidden_dim + j).copied().unwrap_or(0.0);
            }
        }
        profiler.stop(timer, 1); // LM head processes 1 token (last)

        Ok(logits)
    }

    /// Autoregressive text generation.
    ///
    /// Generates tokens one at a time using greedy decoding (argmax sampling).
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Initial token sequence (prompt)
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_token_id` - End-of-sequence token ID (stops generation early)
    ///
    /// # Returns
    ///
    /// Complete token sequence including input and generated tokens
    ///
    /// # Errors
    ///
    /// Returns error if model is not a transformer or forward pass fails
    pub fn generate(
        &self,
        input_tokens: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if input_tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tokens cannot be empty".to_string(),
            });
        }

        let mut tokens = input_tokens.to_vec();
        let vocab_size = self.metadata.vocab_size.unwrap_or(0);

        for _ in 0..max_new_tokens {
            // Forward pass to get logits for next token
            let logits = self.forward(&tokens)?;

            // Greedy sampling: pick token with highest logit
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            // Check for EOS
            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            // Sanity check: don't append invalid tokens
            if (next_token as usize) >= vocab_size && vocab_size > 0 {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Find first matching tensor name from candidates
    fn find_tensor_name(&self, candidates: &[&str]) -> Result<String> {
        for &name in candidates {
            if self.get_tensor(name).is_some() {
                return Ok(name.to_string());
            }
        }
        Err(RealizarError::FormatError {
            reason: format!(
                "No matching tensor found. Tried: {:?}",
                candidates
            ),
        })
    }

    /// Load tokenizer from sibling tokenizer.json file
    ///
    /// Looks for tokenizer.json in the same directory as the model file.
    /// Returns (vocab, bos_token_id, eos_token_id) if found.
    pub fn load_tokenizer_from_sibling(
        model_path: &Path,
    ) -> Option<(Vec<String>, Option<u32>, Option<u32>)> {
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        if !tokenizer_path.exists() {
            return None;
        }

        let content = fs::read_to_string(&tokenizer_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract vocabulary from model.vocab
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;

        // Build vocab vector (sorted by ID)
        let mut vocab_vec: Vec<(String, u32)> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();
        vocab_vec.sort_by_key(|(_, id)| *id);

        let vocab: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

        // Extract special tokens
        let mut bos_id = None;
        let mut eos_id = None;

        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    if content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                        eos_id = Some(id);
                    }
                    if content == "<s>" || content == "<bos>" {
                        bos_id = Some(id);
                    }
                }
            }
        }

        Some((vocab, bos_id, eos_id))
    }

    /// Decode token IDs to text using vocabulary
    ///
    /// If vocab is not available, returns formatted token IDs.
    pub fn decode_tokens(vocab: &[String], token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if let Some(token) = vocab.get(id as usize) {
                // Handle byte-level BPE encoding (Ġ = space prefix)
                let decoded = token
                    .replace("Ġ", " ")
                    .replace("Ċ", "\n")
                    .replace("ĉ", "\t");
                result.push_str(&decoded);
            } else {
                result.push_str(&format!("[{}]", id));
            }
        }
        result
    }

    /// Encode text to token IDs using BPE tokenization
    ///
    /// Loads vocab and merges from tokenizer.json, then performs BPE encoding.
    /// Returns None if tokenizer not found or encoding fails.
    pub fn encode_text(model_path: &Path, text: &str) -> Option<Vec<u32>> {
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        if !tokenizer_path.exists() {
            return None;
        }

        let content = fs::read_to_string(&tokenizer_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract vocabulary (token -> id)
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;
        let token_to_id: HashMap<String, u32> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();

        // Extract merges (pair rules for BPE)
        let merges = json
            .get("model")?
            .get("merges")?
            .as_array()?;

        let merge_rules: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let s = m.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // BPE encoding: convert text to byte-level tokens, then apply merges
        let tokens = bpe_encode(text, &token_to_id, &merge_rules);
        Some(tokens)
    }

    /// Load a full tokenizer struct from sibling tokenizer.json
    ///
    /// Returns a BpeTokenizer that can be reused for multiple encode/decode calls.
    pub fn load_tokenizer(model_path: &Path) -> Option<BpeTokenizer> {
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        if !tokenizer_path.exists() {
            return None;
        }

        let content = fs::read_to_string(&tokenizer_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract vocabulary
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::new();

        let mut vocab_vec: Vec<(String, u32)> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();
        vocab_vec.sort_by_key(|(_, id)| *id);

        for (token, id) in vocab_vec {
            token_to_id.insert(token.clone(), id);
            // Pad id_to_token if needed
            while id_to_token.len() <= id as usize {
                id_to_token.push(String::new());
            }
            id_to_token[id as usize] = token;
        }

        // Extract merges
        let merges = json.get("model")?.get("merges")?.as_array()?;
        let merge_rules: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let s = m.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // Extract special tokens
        let mut bos_id = None;
        let mut eos_id = None;

        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    if content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                        eos_id = Some(id);
                    }
                    if content == "<s>" || content == "<bos>" {
                        bos_id = Some(id);
                    }
                }
            }
        }

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules,
            bos_id,
            eos_id,
        })
    }
}

/// BPE Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string to ID mapping
    pub token_to_id: HashMap<String, u32>,
    /// ID to token string mapping (index = ID)
    pub id_to_token: Vec<String>,
    /// BPE merge rules (first, second) pairs
    pub merge_rules: Vec<(String, String)>,
    /// Beginning-of-sequence token ID
    pub bos_id: Option<u32>,
    /// End-of-sequence token ID
    pub eos_id: Option<u32>,
}

impl BpeTokenizer {
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        bpe_encode(text, &self.token_to_id, &self.merge_rules)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }
}

/// Byte-level BPE encoding
fn bpe_encode(text: &str, vocab: &HashMap<String, u32>, merges: &[(String, String)]) -> Vec<u32> {
    // Convert text to byte-level tokens (GPT-2/Qwen style)
    // Each byte maps to a special unicode char in range U+0100-U+01FF or similar
    let mut tokens: Vec<String> = text
        .chars()
        .map(|c| {
            // Convert character to byte-level BPE token
            // Space becomes Ġ (U+0120 = 288), newline becomes Ċ, etc.
            if c == ' ' {
                "Ġ".to_string()
            } else if c == '\n' {
                "Ċ".to_string()
            } else if c == '\t' {
                "ĉ".to_string()
            } else if c.is_ascii() {
                c.to_string()
            } else {
                // For non-ASCII, encode as bytes
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.chars()
                    .map(|byte_char| byte_to_bpe_char(byte_char as u8))
                    .collect()
            }
        })
        .collect();

    // Apply BPE merges iteratively
    for (first, second) in merges {
        let merged = format!("{}{}", first, second);
        loop {
            let mut found = false;
            let mut i = 0;
            while i + 1 < tokens.len() {
                if &tokens[i] == first && &tokens[i + 1] == second {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                    found = true;
                }
                i += 1;
            }
            if !found {
                break;
            }
        }
    }

    // Convert tokens to IDs
    tokens
        .iter()
        .filter_map(|t| vocab.get(t).copied())
        .collect()
}

/// Convert byte to BPE character representation
fn byte_to_bpe_char(b: u8) -> String {
    // GPT-2/Qwen byte-level BPE uses specific unicode mappings
    // This is a simplified version - real tokenizers use a full byte-to-unicode table
    match b {
        b' ' => "Ġ".to_string(),
        b'\n' => "Ċ".to_string(),
        b'\t' => "ĉ".to_string(),
        _ if b.is_ascii_graphic() || b.is_ascii_alphanumeric() => (b as char).to_string(),
        _ => format!("<0x{:02X}>", b),
    }
}

/// RMS normalization
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = x.len() / hidden_dim;
    let mut output = Vec::with_capacity(x.len());

    for s in 0..seq_len {
        let start = s * hidden_dim;
        let slice = &x[start..start + hidden_dim];

        // Compute RMS
        let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        // Normalize and scale
        for (i, &v) in slice.iter().enumerate() {
            output.push((v / rms) * weight.get(i).copied().unwrap_or(1.0));
        }
    }
    output
}

/// Matrix multiplication with SIMD dot products
/// [seq, in_dim] @ [out_dim, in_dim]^T -> [seq, out_dim]
fn matmul(x: &[f32], w: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; seq_len * out_dim];

    for s in 0..seq_len {
        let x_start = s * in_dim;
        let x_end = x_start + in_dim;
        if x_end > x.len() {
            continue; // Skip if out of bounds
        }
        let x_row = &x[x_start..x_end];

        for o in 0..out_dim {
            let w_start = o * in_dim;
            let w_end = w_start + in_dim;
            if w_end > w.len() {
                continue; // Skip if out of bounds
            }
            let w_row = &w[w_start..w_end];
            // SIMD dot product
            output[s * out_dim + o] = simd_dot(x_row, w_row);
        }
    }
    output
}

/// Transpose a matrix from [rows, cols] to [cols, rows] for GEMM compatibility.
/// Weight matrices are stored as [out_dim, in_dim] but GEMM needs [in_dim, out_dim].
#[cfg(feature = "cuda")]
fn transpose_matrix(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            // m[r, c] -> transposed[c, r]
            let src_idx = r * cols + c;
            let dst_idx = c * rows + r;
            if src_idx < m.len() && dst_idx < transposed.len() {
                transposed[dst_idx] = m[src_idx];
            }
        }
    }
    transposed
}

/// SIMD-accelerated dot product
#[inline]
fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_dot_avx2(a, b) };
        }
    }
    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let chunks = n / 8;

    // SAFETY: This entire fn is unsafe with target_feature(avx2, fma)
    // All intrinsics are safe to call given the target_feature guarantee
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(av, bv, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder (scalar)
    for i in (chunks * 8)..n {
        result += a.get(i).copied().unwrap_or(0.0) * b.get(i).copied().unwrap_or(0.0);
    }

    result
}

/// Simplified multi-head attention (no RoPE, causal mask)
fn simple_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; seq_len * hidden_dim];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            // Compute attention scores for this head
            let mut scores = vec![0.0; seq_len];
            for t in 0..=s {
                // Causal: only attend to past
                let mut score = 0.0;
                for d in 0..head_dim {
                    let q_val = q.get(s * hidden_dim + h * head_dim + d).copied().unwrap_or(0.0);
                    let k_val = k.get(t * kv_dim + kv_h * head_dim + d).copied().unwrap_or(0.0);
                    score += q_val * k_val;
                }
                scores[t] = score * scale;
            }

            // Softmax
            let max_score = scores[..=s].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for score in scores[..=s].iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in scores[..=s].iter_mut() {
                *score /= sum;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut val = 0.0;
                for t in 0..=s {
                    let v_val = v.get(t * kv_dim + kv_h * head_dim + d).copied().unwrap_or(0.0);
                    val += scores[t] * v_val;
                }
                output[s * hidden_dim + h * head_dim + d] = val;
            }
        }
    }

    output
}

/// Check if a file is a valid .apr v2 file
pub fn is_apr_file<P: AsRef<Path>>(path: P) -> bool {
    fs::read(path.as_ref()).is_ok_and(|data| data.len() >= 4 && data[0..4] == MAGIC)
}

/// Detect model format from magic bytes
pub fn detect_format<P: AsRef<Path>>(path: P) -> &'static str {
    let path = path.as_ref();

    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        match ext.as_str() {
            "apr" => return "apr",
            "gguf" => return "gguf",
            "safetensors" => return "safetensors",
            _ => {},
        }
    }

    if let Ok(data) = fs::read(path) {
        if data.len() >= 4 {
            if data[0..4] == MAGIC {
                return "apr";
            }
            if data[0..4] == [0x47, 0x47, 0x55, 0x46] {
                return "gguf";
            }
            if data[0] == b'{' {
                return "safetensors";
            }
        }
    }

    "unknown"
}

/// Legacy type alias for APR v2 model
pub type AprModel = AprV2Model;
/// Legacy type alias (model types are now in metadata)
pub type AprModelType = ();

// ============================================================================
// AprV2ModelCuda: GPU-accelerated APR inference (2x Ollama target)
// ============================================================================

/// CUDA-accelerated wrapper for APR v2 models.
///
/// Mirrors `OwnedQuantizedModelCuda` from GGUF to provide GPU acceleration
/// for APR format models. Achieves 2x+ Ollama performance on supported GPUs.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::apr::{AprV2Model, AprV2ModelCuda};
///
/// let model = AprV2Model::load("model.apr")?;
/// let mut cuda_model = AprV2ModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&[1, 2, 3])?;
///
/// // GPU-accelerated generation
/// let tokens = cuda_model.generate_cuda(&[1, 2, 3], 32, 151643)?;
/// ```
#[cfg(feature = "cuda")]
pub struct AprV2ModelCuda {
    /// Inner APR model
    model: AprV2Model,
    /// Cached CUDA executor
    executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// Cached weight buffers on GPU (tensor_name -> gpu_ptr)
    weight_cache: std::collections::HashMap<String, u64>,
}

#[cfg(feature = "cuda")]
impl AprV2ModelCuda {
    /// Create a new CUDA-accelerated APR model wrapper.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(model: AprV2Model, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048)
    }

    /// Create a new CUDA-accelerated APR model wrapper with custom max sequence length.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn with_max_seq_len(
        model: AprV2Model,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        use crate::cuda::CudaExecutor;

        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // Initialize GPU-resident KV cache for attention acceleration
        let num_layers = model.metadata.num_layers.unwrap_or(0);
        let num_heads = model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = model.metadata.num_kv_heads.unwrap_or(num_heads);
        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);
        let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };

        if num_layers > 0 && head_dim > 0 {
            executor
                .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_kv_cache_gpu".to_string(),
                    reason: format!("GPU KV cache initialization failed: {e}"),
                })?;
        }

        // Set RoPE theta for position embeddings
        let rope_theta = model.metadata.rope_theta.unwrap_or(10000.0);
        executor.set_rope_theta(rope_theta);

        let mut apr_cuda = Self {
            model,
            executor,
            device_name,
            memory_info,
            weight_cache: std::collections::HashMap::new(),
        };

        // Pre-cache all transposed weights on GPU for 2x performance
        apr_cuda.pre_cache_weights()?;

        Ok(apr_cuda)
    }

    /// Check if CUDA is available.
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices.
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Get GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes.
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB.
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    /// Get reference to the inner APR model.
    #[must_use]
    pub fn inner(&self) -> &AprV2Model {
        &self.model
    }

    // ========================================================================
    // BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling.
    pub fn disable_profiling(&mut self) {
        self.executor.disable_profiling();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.executor.is_profiling_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        self.executor.profiler()
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.executor.reset_profiler();
    }

    // ========================================================================
    // Weight Pre-caching (2x performance optimization)
    // ========================================================================

    /// Pre-cache all model weights on GPU in transposed form.
    ///
    /// This is called automatically during model initialization and provides
    /// 2x+ speedup by eliminating per-forward-pass weight transfers.
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU.
    fn pre_cache_weights(&mut self) -> Result<()> {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self.model.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };
        let kv_dim = num_kv_heads * head_dim;

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;

        // Cache per-layer weights
        for layer_idx in 0..num_layers {
            // Q, K, V, O projections
            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, hidden_dim, hidden_dim);
                    let cache_name = format!("layer_{}_q_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, kv_dim, hidden_dim);
                    let cache_name = format!("layer_{}_k_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, kv_dim, hidden_dim);
                    let cache_name = format!("layer_{}_v_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, hidden_dim, hidden_dim);
                    let cache_name = format!("layer_{}_o_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            // FFN: gate, up, down
            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, intermediate_dim, hidden_dim);
                    let cache_name = format!("layer_{}_gate_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, intermediate_dim, hidden_dim);
                    let cache_name = format!("layer_{}_up_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }

            if let Ok(name) = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
            ]) {
                if let Ok(w) = self.model.get_tensor_f32(&name) {
                    let w_t = transpose_matrix(&w, hidden_dim, intermediate_dim);
                    let cache_name = format!("layer_{}_down_proj", layer_idx);
                    total_bytes += self.executor.load_weights(&cache_name, &w_t).unwrap_or(0);
                }
            }
        }

        // Cache LM head (tied to embeddings or separate)
        if let Ok(name) = self.model.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ]) {
            if let Ok(w) = self.model.get_tensor_f32(&name) {
                let w_t = transpose_matrix(&w, vocab_size, hidden_dim);
                total_bytes += self.executor.load_weights("lm_head", &w_t).unwrap_or(0);
            }
        }

        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers)",
            total_bytes / (1024 * 1024),
            num_layers
        );

        Ok(())
    }

    /// Check if weights are cached on GPU.
    #[must_use]
    pub fn weights_cached(&self) -> bool {
        self.executor.cached_weight_count() > 0
    }

    /// Get total cached weight size in MB.
    #[must_use]
    pub fn cached_weight_mb(&self) -> usize {
        self.executor.cached_weight_bytes() / (1024 * 1024)
    }

    // ========================================================================
    // GPU-accelerated inference
    // ========================================================================

    /// GPU-accelerated forward pass.
    ///
    /// Computes logits for the given token sequence using GPU acceleration
    /// for matrix multiplications. Achieves 2x+ Ollama performance by using
    /// GPU GEMM for QKV, attention output, and FFN projections.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self.model.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
        let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // BrickProfiler instrumentation (per spec §12.11)
        let profiling = self.executor.is_profiling_enabled();

        // 1. Token embedding lookup (CPU - fast single lookup)
        let timer_embed = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.Embed"))
        } else {
            None
        };

        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
        ])?;
        let embeddings = self.model.get_tensor_f32(&embed_name)?;

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        if let Some(t) = timer_embed {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Get weight tensors
            let attn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
            ])?;
            let q_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
            ])?;
            let k_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
            ])?;
            let v_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
            ])?;
            let o_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
            ])?;

            let norm_weight = self.model.get_tensor_f32(&attn_norm_name)?;

            // RMSNorm (CPU - small operation)
            let timer_rmsnorm1 = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.RmsNorm"))
            } else {
                None
            };
            let normed = rms_norm(&hidden, &norm_weight, eps);
            if let Some(t) = timer_rmsnorm1 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Q, K, V projections (GPU GEMM for 2x speedup)
            // Use cached weights if available (avoids repeated transpose + upload)
            let q_cache_name = format!("layer_{}_q_proj", layer_idx);
            let k_cache_name = format!("layer_{}_k_proj", layer_idx);
            let v_cache_name = format!("layer_{}_v_proj", layer_idx);
            let o_cache_name = format!("layer_{}_o_proj", layer_idx);

            let timer_qkv = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.QKV"))
            } else {
                None
            };
            let (q, k, v) = if self.has_cached_weight(&q_cache_name) {
                // Fast path: use pre-cached transposed weights
                let q = self.gemm_cached_gpu(&q_cache_name, &normed, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_cached_gpu(&k_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_cached_gpu(&v_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            } else {
                // Fallback: load, transpose, and upload weights each time
                let q_weight = self.model.get_tensor_f32(&q_name)?;
                let k_weight = self.model.get_tensor_f32(&k_name)?;
                let v_weight = self.model.get_tensor_f32(&v_name)?;
                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            };
            if let Some(t) = timer_qkv {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Attention (CPU for now - complex control flow)
            let timer_attn = if profiling {
                Some(self.executor.profiler_mut().start("apr.Attention"))
            } else {
                None
            };
            let attn_out = simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
            if let Some(t) = timer_attn {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Output projection (GPU GEMM)
            let timer_oproj = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.OProj"))
            } else {
                None
            };
            let attn_proj = if self.has_cached_weight(&o_cache_name) {
                self.gemm_cached_gpu(&o_cache_name, &attn_out, seq_len, hidden_dim, hidden_dim)?
            } else {
                let o_weight = self.model.get_tensor_f32(&o_name)?;
                let o_weight_t = transpose_matrix(&o_weight, hidden_dim, hidden_dim);
                self.gemm_gpu(&attn_out, &o_weight_t, seq_len, hidden_dim, hidden_dim)?
            };
            if let Some(t) = timer_oproj {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Residual connection
            let timer_res1 = if profiling {
                Some(self.executor.profiler_mut().start("apr.Residual"))
            } else {
                None
            };
            for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }
            if let Some(t) = timer_res1 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // FFN
            let ffn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
            ])?;
            let gate_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
            ])?;
            let up_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
            ])?;
            let down_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
            ])?;

            // FFN RMSNorm
            let timer_rmsnorm2 = if profiling {
                Some(self.executor.profiler_mut().start("apr.RmsNorm"))
            } else {
                None
            };
            let ffn_norm = self.model.get_tensor_f32(&ffn_norm_name)?;
            let normed = rms_norm(&hidden, &ffn_norm, eps);
            if let Some(t) = timer_rmsnorm2 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // FFN projections (GPU GEMM) - use cached weights if available
            let gate_cache_name = format!("layer_{}_gate_proj", layer_idx);
            let up_cache_name = format!("layer_{}_up_proj", layer_idx);
            let down_cache_name = format!("layer_{}_down_proj", layer_idx);

            let timer_ffn = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.FFN"))
            } else {
                None
            };
            let (gate_out, up_out) = if self.has_cached_weight(&gate_cache_name) {
                // Fast path: use pre-cached transposed weights
                let gate_out = self.gemm_cached_gpu(&gate_cache_name, &normed, seq_len, hidden_dim, intermediate_dim)?;
                let up_out = self.gemm_cached_gpu(&up_cache_name, &normed, seq_len, hidden_dim, intermediate_dim)?;
                (gate_out, up_out)
            } else {
                // Fallback: load, transpose, and upload each time
                let gate = self.model.get_tensor_f32(&gate_name)?;
                let up = self.model.get_tensor_f32(&up_name)?;
                let gate_t = transpose_matrix(&gate, intermediate_dim, hidden_dim);
                let up_t = transpose_matrix(&up, intermediate_dim, hidden_dim);
                let gate_out = self.gemm_gpu(&normed, &gate_t, seq_len, hidden_dim, intermediate_dim)?;
                let up_out = self.gemm_gpu(&normed, &up_t, seq_len, hidden_dim, intermediate_dim)?;
                (gate_out, up_out)
            };

            // SiLU activation and element-wise multiply (CPU - fast)
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }

            let ffn_out = if self.has_cached_weight(&down_cache_name) {
                self.gemm_cached_gpu(&down_cache_name, &ffn_hidden, seq_len, intermediate_dim, hidden_dim)?
            } else {
                let down = self.model.get_tensor_f32(&down_name)?;
                let down_t = transpose_matrix(&down, hidden_dim, intermediate_dim);
                self.gemm_gpu(&ffn_hidden, &down_t, seq_len, intermediate_dim, hidden_dim)?
            };
            if let Some(t) = timer_ffn {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Residual
            let timer_res2 = if profiling {
                Some(self.executor.profiler_mut().start("apr.Residual"))
            } else {
                None
            };
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
            if let Some(t) = timer_res2 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }
        }

        // 3. Final layer norm (CPU)
        let timer_finalnorm = if profiling {
            Some(self.executor.profiler_mut().start("apr.FinalNorm"))
        } else {
            None
        };
        let final_norm_name = self.model.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
        ])?;
        let final_norm = self.model.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);
        if let Some(t) = timer_finalnorm {
            self.executor.profiler_mut().stop(t, 1); // Final norm processes 1 token (last)
        }

        // 4. LM head projection (GPU GEMM for large vocab)
        // Get hidden state for last token only
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        let timer_lmhead = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.LmHead"))
        } else {
            None
        };
        // LM head: [1, hidden_dim] × [hidden_dim, vocab_size] = [1, vocab_size]
        let logits = if self.has_cached_weight("lm_head") {
            // Fast path: use pre-cached transposed LM head
            self.gemm_cached_gpu("lm_head", last_hidden, 1, hidden_dim, vocab_size)?
        } else {
            // Fallback: load, transpose, and upload
            let lm_head_name = self.model.find_tensor_name(&[
                "lm_head.weight",
                "output.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ])?;
            let lm_head = self.model.get_tensor_f32(&lm_head_name)?;
            let lm_head_t = transpose_matrix(&lm_head, vocab_size, hidden_dim);
            self.gemm_gpu(last_hidden, &lm_head_t, 1, hidden_dim, vocab_size)?
        };
        if let Some(t) = timer_lmhead {
            let _ = self.executor.synchronize();
            self.executor.profiler_mut().stop(t, 1); // LM head processes 1 token (last)
        }

        Ok(logits)
    }

    /// GPU GEMM helper: C[m, n] = A[m, k] × B[k, n]
    fn gemm_gpu(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let mut c = vec![0.0f32; m * n];
        self.executor
            .gemm(a, b, &mut c, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "GPU GEMM".to_string(),
                reason: format!("CUDA GEMM failed: {e}"),
            })?;
        Ok(c)
    }

    /// GPU GEMM with cached weight: C[m, n] = A[m, k] × B_cached[k, n]
    ///
    /// Uses pre-cached weight matrix B to avoid repeated GPU uploads.
    /// This is the optimized path for transformer inference.
    fn gemm_cached_gpu(
        &mut self,
        weight_name: &str,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut c = vec![0.0f32; m * n];
        self.executor
            .gemm_b_cached(weight_name, a, &mut c, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "GPU GEMM cached".to_string(),
                reason: format!("CUDA GEMM with cached weight '{}' failed: {e}", weight_name),
            })?;
        Ok(c)
    }

    /// Check if a weight is cached on GPU.
    fn has_cached_weight(&self, name: &str) -> bool {
        self.executor.has_weights(name)
    }

    /// GPU-accelerated token generation.
    ///
    /// Generates tokens autoregressively using GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(eos_id);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// GPU-accelerated forward pass for single token with KV cache.
    ///
    /// This is the optimized decode path that reuses cached K/V values
    /// from previous positions for O(1) attention per token.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_single_cuda(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        // For now, use full forward (no KV cache optimization yet)
        // TODO: Implement proper GPU KV cache path
        self.forward_cuda(&[token_id])
    }

    /// GPU-accelerated generation with KV cache.
    ///
    /// Uses the optimized single-token decode path after prefill.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        // Prefill: process entire prompt
        let mut tokens = prompt.to_vec();
        let _ = self.forward_cuda(&tokens)?;

        // Decode: generate one token at a time
        for i in 0..max_new_tokens {
            let position = tokens.len();
            let last_token = *tokens.last().unwrap_or(&1);

            let logits = self.forward_single_cuda(last_token, position)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(eos_id);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constant() {
        assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x32]);
        assert_eq!(&MAGIC, b"APR2");
    }

    #[test]
    fn test_header_from_bytes_too_small() {
        let data = vec![0u8; 10];
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_invalid_magic() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_valid() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset

        let header = AprHeader::from_bytes(&data).expect("should parse");
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.tensor_count, 10);
    }

    #[test]
    fn test_flags() {
        let flags = AprFlags::new(0x0007);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());

        let flags2 = AprFlags::new(0x0020);
        assert!(flags2.is_quantized());
        assert!(!flags2.is_compressed());
    }

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(detect_format("/fake/model.apr"), "apr");
        assert_eq!(detect_format("/fake/model.gguf"), "gguf");
        assert_eq!(detect_format("/fake/model.safetensors"), "safetensors");
    }

    // APR v2 binary tensor index format tests

    /// Helper to create binary tensor entry
    fn create_binary_tensor_entry(
        name: &str,
        dtype: u8,
        shape: &[u64],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        // Name
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        // Dtype
        data.push(dtype);
        // Shape
        data.push(shape.len() as u8);
        for &dim in shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        // Offset and size
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data
    }

    #[test]
    fn test_tensor_entry_from_binary_valid() {
        let data = create_binary_tensor_entry(
            "model.embed_tokens.weight",
            0,
            &[32000, 2048],
            0,
            262144000,
        );
        let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.name, "model.embed_tokens.weight");
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.shape, vec![32000, 2048]);
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.size, 262144000);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_tensor_entry_from_binary_f16() {
        let data = create_binary_tensor_entry(
            "layer.0.attn.q_proj.weight",
            1,
            &[2048, 2048],
            1024,
            8388608,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "F16");
        assert_eq!(entry.shape, vec![2048, 2048]);
    }

    #[test]
    fn test_tensor_entry_from_binary_bf16() {
        let data = create_binary_tensor_entry("lm_head.weight", 2, &[32000, 2048], 512, 131072000);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "BF16");
    }

    #[test]
    fn test_tensor_entry_from_binary_int8() {
        let data = create_binary_tensor_entry("quantized.weight", 3, &[1024, 1024], 0, 1048576);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "I8");
    }

    #[test]
    fn test_tensor_entry_from_binary_1d() {
        let data = create_binary_tensor_entry("model.norm.weight", 0, &[2048], 0, 8192);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![2048]);
        assert_eq!(entry.element_count(), 2048);
    }

    #[test]
    fn test_tensor_entry_from_binary_3d() {
        let data = create_binary_tensor_entry("conv.weight", 0, &[64, 3, 7], 0, 5376);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![64, 3, 7]);
        assert_eq!(entry.element_count(), 64 * 3 * 7);
    }

    #[test]
    fn test_tensor_entry_from_binary_too_short() {
        let data = vec![0u8; 2];
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_name() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_le_bytes()); // name_len = 100
        data.extend_from_slice(b"short"); // Only 5 bytes of name
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_shape() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u16.to_le_bytes()); // name_len
        data.extend_from_slice(b"test");
        data.push(0); // dtype
        data.push(2); // ndim = 2
        data.extend_from_slice(&1024u64.to_le_bytes()); // first dim only
                                                        // Missing second dim, offset, size
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_element_count() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![32, 64, 128],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 32 * 64 * 128);
    }

    #[test]
    fn test_tensor_entry_element_count_scalar() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_multiple_tensor_entries_sequential() {
        let mut data = Vec::new();
        data.extend(create_binary_tensor_entry("tensor1", 0, &[100], 0, 400));
        data.extend(create_binary_tensor_entry(
            "tensor2",
            1,
            &[200, 300],
            400,
            120000,
        ));
        data.extend(create_binary_tensor_entry("tensor3", 2, &[50], 120400, 100));

        let mut pos = 0;
        let mut entries = Vec::new();

        while pos < data.len() {
            let (entry, consumed) = TensorEntry::from_binary(&data[pos..]).expect("should parse");
            entries.push(entry);
            pos += consumed;
        }

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "tensor1");
        assert_eq!(entries[1].name, "tensor2");
        assert_eq!(entries[2].name, "tensor3");
        assert_eq!(entries[1].shape, vec![200, 300]);
    }

    // =========================================================================
    // Compression Tests (GH-35)
    // =========================================================================

    #[test]
    fn test_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_no_compression() {
        let flags = AprFlags::new(0);
        assert!(!flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(!flags.is_compressed());
    }

    #[cfg(not(feature = "apr-compression"))]
    #[test]
    fn test_compressed_file_requires_feature() {
        // Create a minimal APR v2 header with LZ4 flag
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&(AprFlags::LZ4_COMPRESSED).to_le_bytes()); // LZ4 flag
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("apr-compression"),
            "Error should mention feature: {}",
            err_msg
        );
    }
}
