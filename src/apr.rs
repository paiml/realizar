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
    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Additional metadata fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
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

        if header.flags.is_compressed() {
            return Err(RealizarError::FormatError {
                reason: "Compressed .apr files not yet supported".to_string(),
            });
        }

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
}
