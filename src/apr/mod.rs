//! Aprender .apr format support for realizar (APR v2 only)
//!
//! This module provides loading and inference for models in Aprender's native
//! .apr v2 format (Magic: `APR\0` = 0x41505232).
//!
//! ## Format Structure (APR v2, 64-byte header)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                           │
//! │   - Magic: "APR\0" (4 bytes)                                 │
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
use std::fs::{self, File};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

// PMAT-802: Extracted modules
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
mod dequant;
mod helpers;
mod model_data;
mod tokenizer;

// PMAT-COMPLY: Re-export ModelData
pub use model_data::ModelData;

// PMAT-COMPLY: Re-export dequantization functions
pub(crate) use dequant::{dequantize_f16, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0};
// Re-export for tests
pub use dequant::{dtype_to_ggml_qtype, f16_to_f32, is_quantized_dtype};

// T-COV-95: Test factory for Active Pygmy APR models
#[cfg(test)]
pub(crate) mod test_factory;

#[cfg(feature = "cuda")]
pub use cuda::AprV2ModelCuda;
#[cfg(feature = "cuda")]
use helpers::apply_rope_norm;
#[cfg(feature = "cuda")]
use helpers::transpose_matrix;
pub use helpers::{detect_format, is_apr_file, simd_dot};
use helpers::{matmul, rms_norm, simple_attention};
use tokenizer::bpe_encode;
pub use tokenizer::{byte_to_bpe_char, BpeTokenizer, SimpleTokenizer};

/// Magic number: "APR" followed by version byte
/// - Legacy: APR\0 (0x41, 0x50, 0x52, 0x00)
/// - v1: APR1 (0x41, 0x50, 0x52, 0x31)
/// - v2: APR2 (0x41, 0x50, 0x52, 0x32)
pub const MAGIC_PREFIX: [u8; 3] = [0x41, 0x50, 0x52]; // "APR"

/// Legacy magic for compatibility
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00];

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
    /// Magic number ("APR\0")
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

        // Check magic - first 3 bytes must be "APR", 4th byte is version
        let magic: [u8; 4] = data[0..4]
            .try_into()
            .map_err(|_| RealizarError::FormatError {
                reason: "Failed to read magic bytes".to_string(),
            })?;

        // Validate magic prefix (APR)
        if magic[0..3] != MAGIC_PREFIX {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid .apr magic: expected APR {:?}, got {:?}",
                    MAGIC_PREFIX,
                    &magic[0..3]
                ),
            });
        }

        // Validate version byte (0, '1', or '2')
        let version_byte = magic[3];
        if version_byte != 0 && version_byte != b'1' && version_byte != b'2' {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid .apr version byte: expected 0, '1', or '2', got {}",
                    version_byte
                ),
            });
        }

        // APR v1 (magic "APR1") has different header layout - not supported for inference
        // APR v1 is used by Whisper models but has inline tensor index format
        if version_byte == b'1' {
            return Err(RealizarError::UnsupportedOperation {
                operation: "load_apr_v1".to_string(),
                reason: "APR v1 format not supported for inference. \
                        Use 'apr convert model.apr -o model_v2.apr --format apr2' \
                        to convert to APR v2 format, or use the GGUF version."
                    .to_string(),
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
        // GH-191 FIX: Writers now use GGML dtype values directly.
        // This reader must map GGML type IDs to canonical dtype strings.
        // GGML types: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1,
        //   8=Q8_0, 9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 30=BF16
        let dtype_byte = data[pos];
        pos += 1;
        let dtype = match dtype_byte {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            30 => "BF16",
            _ => {
                eprintln!("WARN: Unknown APR dtype byte {dtype_byte}, treating as F32");
                "F32"
            },
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
///
/// # Schema Resilience (PMAT-111)
///
/// This struct uses serde aliases to accept multiple field name variations
/// commonly found in different model formats (HuggingFace, GGUF, etc.):
///
/// - `hidden_size` also accepts: `hidden_dim`, `d_model`, `n_embd`
/// - `num_layers` also accepts: `n_layers`, `num_hidden_layers`, `n_layer`
/// - `num_heads` also accepts: `n_heads`, `num_attention_heads`, `n_head`
/// - `vocab_size` also accepts: `n_vocab`
/// - `intermediate_size` also accepts: `ffn_dim`, `intermediate_dim`, `n_inner`
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
    /// Aliases: hidden_dim, d_model, n_embd (PMAT-111 schema resilience)
    #[serde(default, alias = "hidden_dim", alias = "d_model", alias = "n_embd")]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    /// Aliases: n_layers, num_hidden_layers, n_layer (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "n_layers",
        alias = "num_hidden_layers",
        alias = "n_layer"
    )]
    pub num_layers: Option<usize>,
    /// Number of attention heads
    /// Aliases: n_heads, num_attention_heads, n_head (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "n_heads",
        alias = "num_attention_heads",
        alias = "n_head"
    )]
    pub num_heads: Option<usize>,
    /// Number of key-value heads (for GQA, defaults to num_heads)
    /// Aliases: n_kv_heads (PMAT-111 schema resilience)
    #[serde(default, alias = "n_kv_heads")]
    pub num_kv_heads: Option<usize>,
    /// Vocabulary size
    /// Aliases: n_vocab (PMAT-111 schema resilience)
    #[serde(default, alias = "n_vocab")]
    pub vocab_size: Option<usize>,
    /// FFN intermediate dimension
    /// Aliases: ffn_dim, intermediate_dim, n_inner (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "ffn_dim",
        alias = "intermediate_dim",
        alias = "n_inner"
    )]
    pub intermediate_size: Option<usize>,
    /// Maximum context/sequence length
    /// Aliases: max_seq_len, context_length, n_ctx (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "max_seq_len",
        alias = "context_length",
        alias = "n_ctx"
    )]
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta for position encoding
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default)]
    pub rope_type: Option<u32>,
    /// Layer norm epsilon
    /// Aliases: layer_norm_eps, norm_eps (PMAT-111 schema resilience)
    #[serde(default, alias = "layer_norm_eps", alias = "norm_eps")]
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

    /// Extract embedded tokenizer vocabulary from APR metadata (GH-156)
    ///
    /// APR files can contain embedded tokenizer data in the metadata section.
    /// This is the preferred way to decode tokens - no sibling files needed.
    ///
    /// # Returns
    /// - `Some(vocab)` if tokenizer.vocabulary is present in metadata
    /// - `None` if no embedded tokenizer
    #[must_use]
    pub fn get_embedded_vocabulary(&self) -> Option<Vec<String>> {
        let vocab_value = self.extra.get("tokenizer.vocabulary")?;
        let vocab_array = vocab_value.as_array()?;

        let vocab: Vec<String> = vocab_array
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if vocab.is_empty() {
            None
        } else {
            Some(vocab)
        }
    }

    /// Get embedded BOS token ID from APR metadata
    #[must_use]
    pub fn get_embedded_bos_token_id(&self) -> Option<u32> {
        self.extra
            .get("tokenizer.bos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
    }

    /// Get embedded EOS token ID from APR metadata
    #[must_use]
    pub fn get_embedded_eos_token_id(&self) -> Option<u32> {
        self.extra
            .get("tokenizer.eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
    }

    /// Extract embedded BPE merge rules from APR metadata (PMAT-171)
    ///
    /// APR files converted from GGUF can contain embedded BPE merge rules
    /// for standalone encoding - no sibling tokenizer.json needed.
    ///
    /// # Returns
    /// - `Some(merges)` if tokenizer.merges is present in metadata
    /// - `None` if no embedded merges
    #[must_use]
    pub fn get_embedded_merges(&self) -> Option<Vec<(String, String)>> {
        let merges_value = self.extra.get("tokenizer.merges")?;
        let merges_array = merges_value.as_array()?;

        let merges: Vec<(String, String)> = merges_array
            .iter()
            .filter_map(|v| {
                let s = v.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        if merges.is_empty() {
            None
        } else {
            Some(merges)
        }
    }
}

/// APR v2 model for realizar inference
///
/// # Memory Management
///
/// Uses memory-mapped I/O for uncompressed files to avoid zram pressure.
/// After loading tensors to GPU, call `release_cpu_pages()` to advise
/// the kernel that pages can be dropped (re-faulted from disk if needed).
///
/// # References
///
/// - Didona et al. (2022): mmap vs read() performance
/// - See docs/model-loading.md for full design rationale
#[derive(Debug)]
pub struct AprV2Model {
    /// Header information
    header: AprHeader,
    /// Model metadata
    metadata: AprMetadata,
    /// Tensor index
    tensors: Vec<TensorEntry>,
    /// Raw file data (mmap for uncompressed, heap for compressed)
    data: ModelData,
}

impl AprV2Model {
    /// Load a model from a .apr file using memory mapping.
    ///
    /// # Memory Efficiency
    ///
    /// For uncompressed files, uses `mmap()` for zero-copy access.
    /// The kernel manages pages via demand paging - only accessed
    /// pages are loaded into RAM. After GPU transfer, call
    /// `release_cpu_pages()` to advise the kernel to drop pages.
    ///
    /// For compressed files, falls back to heap allocation after
    /// decompression (mmap not possible for decompressed data).
    ///
    /// # References
    ///
    /// - Didona et al. (2022): mmap achieves 2.3x throughput vs read()
    /// - See docs/model-loading.md for design rationale
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let path_ref = path.as_ref();

        // Read just the header first to check for compression
        let mut file = File::open(path_ref).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open .apr file: {e}"),
        })?;

        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to read .apr header: {e}"),
            })?;

        let header = AprHeader::from_bytes(&header_buf)?;

        // Check for unsupported features
        if header.flags.is_encrypted() {
            return Err(RealizarError::FormatError {
                reason: "Encrypted .apr files not yet supported".to_string(),
            });
        }

        // Choose loading strategy based on compression
        let data = if header.flags.is_compressed() {
            // Compressed: must read entire file into heap, then decompress
            drop(file); // Close file handle
            let raw_data = std::fs::read(path_ref).map_err(|e| RealizarError::IoError {
                message: format!("Failed to read compressed .apr file: {e}"),
            })?;
            let decompressed = Self::decompress_apr_data(&header, raw_data)?;
            ModelData::from_vec(decompressed)
        } else {
            // Uncompressed: use mmap for zero-copy access
            drop(file); // Close file handle before mmap
            ModelData::open_mmap(path_ref)?
        };

        // Advise sequential access pattern for parsing
        #[cfg(unix)]
        let _ = data.advise_sequential();

        Self::from_model_data(header, data)
    }

    /// Load a model from a .apr file (WASM fallback).
    #[cfg(target_arch = "wasm32")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw_data = std::fs::read(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read .apr file: {e}"),
        })?;
        Self::from_bytes(raw_data)
    }

    /// Load a model from bytes (heap-allocated).
    ///
    /// Use this for:
    /// - Compressed files after decompression
    /// - Data received over network
    /// - WASM environments (no mmap support)
    ///
    /// For file-based loading with mmap support, use `load()` instead.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        // Parse header
        let header = AprHeader::from_bytes(&data)?;

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

        Self::from_model_data(header, ModelData::from_vec(data))
    }

    /// Internal: construct model from header and ModelData.
    fn from_model_data(header: AprHeader, data: ModelData) -> Result<Self> {
        let data_slice = data.as_slice();

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data_slice.len() < metadata_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr file truncated: metadata extends to {} but file is {} bytes",
                    metadata_end,
                    data_slice.len()
                ),
            });
        }

        let metadata: AprMetadata = if header.metadata_size > 0 {
            serde_json::from_slice(&data_slice[metadata_start..metadata_end]).unwrap_or_default()
        } else {
            AprMetadata::default()
        };

        // Parse tensor index (binary format from aprender v2)
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        if index_start < index_end && index_end <= data_slice.len() {
            let index_data = &data_slice[index_start..index_end];
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
            Ok(result)
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
        let data_slice = self.data.as_slice();

        if end > data_slice.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        let bytes = &data_slice[start..end];

        // Calculate total number of elements from shape
        let num_elements: usize = entry.shape.iter().product();

        // Parse based on dtype
        match entry.dtype.as_str() {
            "F32" | "f32" => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            },
            "F16" | "f16" => Ok(dequantize_f16(bytes, num_elements)),
            "Q8_0" | "q8_0" => Ok(dequantize_q8_0(bytes, num_elements)),
            "Q4_K" | "q4_k" => Ok(dequantize_q4_k(bytes, num_elements)),
            "Q6_K" | "q6_k" => Ok(dequantize_q6_k(bytes, num_elements)),
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
        let data_slice = self.data.as_slice();

        if end > data_slice.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        Ok(&data_slice[start..end])
    }

    /// Release CPU pages after GPU transfer (Unix only).
    ///
    /// Advises the kernel that the mapped pages are no longer needed.
    /// The kernel will drop pages immediately (not compress to zram)
    /// and re-fault from disk if accessed again.
    ///
    /// # When to Call
    ///
    /// After all tensor data has been copied to GPU via `cuMemcpy()`.
    /// This is the key method for reducing zram pressure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = AprV2Model::load("model.apr")?;
    /// for name in model.tensor_names() {
    ///     let bytes = model.get_tensor_bytes(&name)?;
    ///     cuda::memcpy_htod(gpu_ptr, bytes);
    /// }
    /// // Free CPU pages now that data is on GPU
    /// model.release_cpu_pages()?;
    /// ```
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn release_cpu_pages(&self) -> Result<()> {
        self.data.release_cpu_pages()
    }

    /// Check if model is using memory-mapped I/O.
    ///
    /// Returns `true` if the model was loaded via mmap (uncompressed file).
    /// Returns `false` if the model is heap-allocated (compressed file or WASM).
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        self.data.is_mmap()
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
        let intermediate_dim = self.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
        let eps = self.metadata.rms_norm_eps.unwrap_or(1e-6);

        // 1. Token embedding lookup
        let embed_name = self.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight", // SafeTensors (no model. prefix)
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight", // GGUF naming convention
        ])?;

        let embeddings = self.get_tensor_f32(&embed_name)?;
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Try common naming patterns (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF naming
            ])?;

            let q_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
                &format!("blk.{layer_idx}.attn_q.weight"), // GGUF naming
            ])?;

            let k_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
                &format!("blk.{layer_idx}.attn_k.weight"), // GGUF naming
            ])?;

            let v_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
                &format!("blk.{layer_idx}.attn_v.weight"), // GGUF naming
            ])?;

            let o_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF naming
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
            let k = matmul(
                &normed,
                &k_weight,
                seq_len,
                hidden_dim,
                num_kv_heads * head_dim,
            );
            let v = matmul(
                &normed,
                &v_weight,
                seq_len,
                hidden_dim,
                num_kv_heads * head_dim,
            );

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
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
                &format!("blk.{layer_idx}.ffn_norm.weight"), // GGUF naming
            ])?;

            let gate_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"), // GGUF naming
            ])?;

            let up_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"), // GGUF naming
            ])?;

            let down_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"), // SafeTensors
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"), // GGUF naming
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
            "norm.weight", // SafeTensors
            "transformer.ln_f.weight",
            "output_norm.weight", // GGUF naming
        ])?;
        let final_norm = self.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);

        // 4. LM head (last token only for generation)
        // BUG-APR-001: Add token_embd.weight for GGUF weight tying
        let lm_head_name = self.find_tensor_name(&[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight", // Tied embeddings
            "embed_tokens.weight",       // SafeTensors tied embeddings
            "token_embd.weight",         // GGUF tied embeddings
        ])?;
        let lm_head = self.get_tensor_f32(&lm_head_name)?;

        // Get hidden state for last token
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        // BUG-APR-001-FIX: Detect weight tying and handle transposed layout
        // GGUF token_embd.weight is stored as [hidden_dim, vocab_size] (column-major)
        // Regular lm_head.weight is stored as [vocab_size, hidden_dim] (row-major)
        // When using tied embeddings, we need to transpose the access pattern.
        let is_tied_embedding =
            lm_head_name == "token_embd.weight" || lm_head_name.ends_with("embed_tokens.weight");

        // Project to vocab
        let mut logits = vec![0.0; vocab_size];
        if is_tied_embedding && lm_head.len() == hidden_dim * vocab_size {
            // Tied embedding: token_embd is [hidden_dim, vocab_size]
            // Access: logit[i] = sum_j(hidden[j] * embed[j * vocab_size + i])
            for (i, logit) in logits.iter_mut().enumerate() {
                for (j, &h) in last_hidden.iter().enumerate() {
                    *logit += h * lm_head.get(j * vocab_size + i).copied().unwrap_or(0.0);
                }
            }
        } else {
            // Regular lm_head: [vocab_size, hidden_dim]
            // Access: logit[i] = sum_j(hidden[j] * lm_head[i * hidden_dim + j])
            for (i, logit) in logits.iter_mut().enumerate() {
                for (j, &h) in last_hidden.iter().enumerate() {
                    *logit += h * lm_head.get(i * hidden_dim + j).copied().unwrap_or(0.0);
                }
            }
        }

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
                .map_or(0, |(idx, _)| idx as u32);

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
            reason: format!("No matching tensor found. Tried: {:?}", candidates),
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
                let id = token
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32);

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

    /// Encode text to token IDs using embedded BPE tokenizer (PMAT-172: Fail-Fast)
    ///
    /// APR files MUST have embedded tokenizer. NO FALLBACK to external files.
    /// This prevents Silent Failure Recovery where wrong tokenizer produces garbage.
    ///
    /// # Design Principle
    ///
    /// APR format is designed to be ONE self-contained file. If the embedded
    /// tokenizer is missing, the APR file is BROKEN and should be re-converted.
    ///
    /// # Returns
    ///
    /// - `Some(tokens)` if APR has embedded tokenizer
    /// - `None` if file doesn't exist or isn't APR format
    ///
    /// # Panics
    ///
    /// Prints error and returns None if APR is missing embedded tokenizer.
    /// This is intentional - we want users to see the error, not garbage output.
    pub fn encode_text(model_path: &Path, text: &str) -> Option<Vec<u32>> {
        // Validate model path exists
        if !model_path.exists() {
            eprintln!(
                "[PMAT-172] Error: Model file not found: {}",
                model_path.display()
            );
            return None;
        }

        // PMAT-172: APR files MUST use embedded tokenizer - NO FALLBACK
        if model_path.extension().is_some_and(|e| e == "apr") {
            match Self::load(model_path) {
                Ok(model) => {
                    match model.load_embedded_bpe_tokenizer() {
                        Some(tokenizer) => {
                            return Some(tokenizer.encode(text));
                        },
                        None => {
                            // PMAT-172: FAIL FAST - Do not fall back to external tokenizers
                            eprintln!("\n[PMAT-172] ERROR: APR file missing embedded tokenizer.");
                            eprintln!("           APR format requires self-contained tokenizer.");
                            eprintln!(
                                "           Re-convert with: apr convert <source>.gguf -o {}",
                                model_path.display()
                            );
                            eprintln!("           Or use the original GGUF file directly.\n");
                            return None;
                        },
                    }
                },
                Err(e) => {
                    eprintln!("[PMAT-172] Error loading APR file: {}", e);
                    return None;
                },
            }
        }

        // For non-APR files (SafeTensors), use sibling tokenizer.json ONLY
        // NO fallback to HuggingFace cache (PMAT-172: removed Silent Failure Recovery)
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!(
                "\n[PMAT-172] ERROR: No tokenizer found for {}.",
                model_path.display()
            );
            eprintln!(
                "           Expected sibling file: {}",
                tokenizer_path.display()
            );
            eprintln!(
                "           For SafeTensors models, tokenizer.json must be in same directory.\n"
            );
            return None;
        }

        let content = match fs::read_to_string(&tokenizer_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[PMAT-172] Error reading tokenizer.json: {}", e);
                return None;
            },
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[PMAT-172] Error parsing tokenizer.json: {}", e);
                return None;
            },
        };

        // Extract vocabulary (token -> id)
        let vocab_obj = json.get("model")?.get("vocab")?;
        let vocab_map = vocab_obj.as_object()?;
        let token_to_id: HashMap<String, u32> = vocab_map
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
            .collect();

        // F-REGR-231: Extract added_tokens (special tokens like <|im_start|>, <|im_end|>)
        let special_tokens: HashMap<String, u32> = json
            .get("added_tokens")
            .and_then(|arr| arr.as_array())
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|t| {
                        let content = t.get("content")?.as_str()?;
                        let id = t.get("id")?.as_u64()? as u32;
                        Some((content.to_string(), id))
                    })
                    .collect()
            })
            .unwrap_or_default();

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

        let tokens = bpe_encode(text, &token_to_id, &merge_rules, &special_tokens);
        Some(tokens)
    }

    // PMAT-172: Removed find_tokenizer_json_in_cache() — loading a stale
    // HuggingFace cache tokenizer produced garbage output. Now requires
    // explicit tokenizer path or embedded APR vocabulary.

    /// Load tokenizer from embedded APR metadata (GH-156)
    ///
    /// APR files can contain embedded tokenizer data - this is the preferred
    /// way to decode tokens since it doesn't require sibling files.
    ///
    /// Returns a simple decode-only tokenizer (no BPE encoding support).
    pub fn load_embedded_tokenizer(&self) -> Option<SimpleTokenizer> {
        let vocab = self.metadata.get_embedded_vocabulary()?;
        let bos_id = self.metadata.get_embedded_bos_token_id();
        let eos_id = self.metadata.get_embedded_eos_token_id();

        Some(SimpleTokenizer {
            id_to_token: vocab,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        })
    }

    /// Load a full BPE tokenizer from embedded APR metadata (PMAT-171)
    ///
    /// APR files converted from GGUF can contain both vocabulary AND BPE merge
    /// rules embedded in metadata. This enables standalone encoding without
    /// needing sibling tokenizer.json files.
    ///
    /// Returns `Some(BpeTokenizer)` if both vocab and merges are embedded.
    /// Returns `None` if either is missing (fall back to sibling file).
    pub fn load_embedded_bpe_tokenizer(&self) -> Option<BpeTokenizer> {
        let vocab_list = self.metadata.get_embedded_vocabulary()?;
        let merges = self.metadata.get_embedded_merges()?;

        // Build token_to_id and id_to_token maps
        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::with_capacity(vocab_list.len());

        for (id, token) in vocab_list.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
            id_to_token.push(token.clone());
        }

        let bos_id = self.metadata.get_embedded_bos_token_id();
        let eos_id = self.metadata.get_embedded_eos_token_id();

        // GH-189: Extract special tokens from vocabulary for atomic tokenization
        // Special tokens like <|im_start|>, <|im_end|> must not be split by BPE
        let special_tokens = extract_special_tokens_from_vocab(&token_to_id);

        eprintln!(
            "[PMAT-171] Loaded embedded BPE tokenizer: {} vocab, {} merges, {} special tokens",
            id_to_token.len(),
            merges.len(),
            special_tokens.len()
        );

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: merges,
            bos_id,
            eos_id,
            special_tokens,
        })
    }

    /// Load a full tokenizer struct from sibling tokenizer.json
    ///
    /// Returns a BpeTokenizer that can be reused for multiple encode/decode calls.
    /// For decode-only operations, prefer `load_embedded_tokenizer()` first.
    pub fn load_tokenizer(model_path: &Path) -> Option<BpeTokenizer> {
        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        Self::load_tokenizer_from_path(&tokenizer_path)
    }

    /// Load a BPE tokenizer from an explicit tokenizer.json path
    ///
    /// This is used for loading tokenizers from HuggingFace cache or other locations.
    /// (PMAT-SHOWCASE-TOKENIZER-001)
    pub fn load_tokenizer_from_path(tokenizer_path: &Path) -> Option<BpeTokenizer> {
        if !tokenizer_path.exists() {
            return None;
        }

        let content = fs::read_to_string(tokenizer_path).ok()?;
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

        // GH-189: Extract ALL added_tokens as special tokens for atomic tokenization
        let mut bos_id = None;
        let mut eos_id = None;
        let mut special_tokens: HashMap<String, u32> = HashMap::new();

        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    // Add ALL added_tokens to special_tokens map for atomic tokenization
                    special_tokens.insert(content.to_string(), id);

                    // Also track bos/eos specifically
                    if content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                        eos_id = Some(id);
                    }
                    if content == "<s>" || content == "<bos>" {
                        bos_id = Some(id);
                    }
                }
            }
        }

        eprintln!(
            "[GH-189] Loaded tokenizer from {}: {} special tokens",
            tokenizer_path.display(),
            special_tokens.len()
        );

        Some(BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules,
            bos_id,
            eos_id,
            special_tokens,
        })
    }
}

/// Legacy type alias for APR v2 model
pub type AprModel = AprV2Model;
/// Legacy type alias (model types are now in metadata)
pub type AprModelType = ();

/// Extract special tokens from vocabulary for atomic tokenization (GH-189)
///
/// Special tokens like `<|im_start|>`, `<|im_end|>` must be tokenized atomically,
/// not split character-by-character. This function scans the vocabulary for
/// common special token patterns used by Qwen, ChatML, LLaMA, and other models.
#[allow(clippy::implicit_hasher)]
pub fn extract_special_tokens_from_vocab(
    token_to_id: &HashMap<String, u32>,
) -> HashMap<String, u32> {
    let mut special_tokens = HashMap::new();

    // Common special token patterns across model families
    let patterns = [
        // ChatML / Qwen style
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        // LLaMA / general
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        // Phi / Mistral style
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|end|>",
        // Code models
        "<|file_separator|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|repo_name|>",
        // Additional Qwen tokens
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
    ];

    for pattern in patterns {
        if let Some(&id) = token_to_id.get(pattern) {
            special_tokens.insert(pattern.to_string(), id);
        }
    }

    // Also scan for any token matching <|...|> pattern (common for special tokens)
    for (token, &id) in token_to_id {
        if token.starts_with("<|") && token.ends_with("|>") && !special_tokens.contains_key(token) {
            special_tokens.insert(token.clone(), id);
        }
    }

    special_tokens
}

// =============================================================================

use memmap2::Mmap;

/// Memory-mapped APR model for fast loading and GPU inference
///
/// Similar to MappedGGUFModel, this provides zero-copy access to APR tensor data.
/// The file is memory-mapped for fast startup (~36x faster than full file read).
#[derive(Debug)]
pub struct MappedAprModel {
    /// APR header
    pub header: AprHeader,
    /// Model metadata
    pub metadata: AprMetadata,
    /// Tensor index
    pub tensors: Vec<TensorEntry>,
    /// Memory-mapped file data
    mmap: Mmap,
}

impl MappedAprModel {
    /// Load an APR model with memory mapping for fast startup
    ///
    /// # Arguments
    /// * `path` - Path to the .apr file
    ///
    /// # Errors
    /// Returns error if file cannot be opened or has invalid format.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open .apr file: {e}"),
        })?;

        // SAFETY: File is opened read-only, callers validate format before trusting data
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::IoError {
                message: format!("Failed to mmap .apr file: {e}"),
            })?
        };

        Self::from_mmap(mmap)
    }

    /// Create from existing memory map
    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let data = &mmap[..];

        // Parse header
        let header = AprHeader::from_bytes(data)?;

        // Validate magic
        if header.magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: "Invalid APR magic bytes".to_string(),
            });
        }

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data.len() < metadata_end {
            return Err(RealizarError::FormatError {
                reason: "APR file truncated: metadata extends past EOF".to_string(),
            });
        }

        let metadata: AprMetadata = if header.metadata_size > 0 {
            serde_json::from_slice(&data[metadata_start..metadata_end]).unwrap_or_default()
        } else {
            AprMetadata::default()
        };

        // Parse tensor index
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
                    Err(_) => break,
                }
            }
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            mmap,
        })
    }

    /// Get raw file data (for tensor access)
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Get file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get tensor count
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get data offset (start of tensor data section)
    #[must_use]
    pub fn data_offset(&self) -> u64 {
        self.header.data_offset
    }

    /// Find tensor by name
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get raw tensor data by name
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = self.header.data_offset as usize + tensor.offset as usize;
        let end = start + tensor.size as usize;

        if end > self.mmap.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor {name} extends past EOF"),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Convert APR dtype string to GGML qtype
    #[must_use]
    pub fn dtype_to_qtype(dtype: &str) -> u32 {
        match dtype {
            "F32" => 0,
            "F16" => 1,
            "Q4_0" => 2,
            "Q4_1" => 3,
            "Q5_0" => 6,
            "Q5_1" => 7,
            "Q8_0" => 8,
            "Q8_1" => 9,
            "Q2_K" => 10,
            "Q3_K" => 11,
            "Q4_K" => 12,
            "Q5_K" => 13,
            "Q6_K" => 14,
            "IQ2_XXS" => 16,
            "IQ2_XS" => 17,
            "BF16" => 30,
            _ => 0, // Default to F32
        }
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod apr_tests;

// Additional tests for coverage (tests_part_02.rs)
#[cfg(test)]
#[path = "tests_part_02.rs"]
mod apr_tests_part_02;

// T-COV-95 Additional coverage (tests_part_03.rs)
#[cfg(test)]
#[path = "tests_part_03.rs"]
mod apr_tests_part_03;

// T-COV-95 Active APR Pygmy: Cross-Format Dynamic Falsification (tests_part_04.rs)
#[cfg(test)]
#[path = "tests_part_04.rs"]
mod apr_tests_part_04;
