//! Safetensors parser
//!
//! Pure Rust implementation of Safetensors format reader.
//! Used by `HuggingFace` for safe, zero-copy tensor storage.
//!
//! Format specification: <https://github.com/huggingface/safetensors>
//!
//! ## Format Overview
//!
//! ```text
//! Safetensors := HEADER METADATA TENSOR_DATA
//!
//! HEADER := {
//!   metadata_len: u64 (little-endian)
//! }
//!
//! METADATA := JSON {
//!   "tensor_name": {
//!     "dtype": "F32" | "F16" | "I32" | ...,
//!     "shape": [dim1, dim2, ...],
//!     "data_offsets": [start, end]
//!   },
//!   ...
//! }
//! ```

use std::{
    collections::HashMap,
    io::{Cursor, Read},
};

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};
use crate::inference::simd_bf16_to_f32;

/// Safetensors data type
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum SafetensorsDtype {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// Brain float 16
    BF16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

/// JSON tensor metadata (internal)
#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: SafetensorsDtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Tensor metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: SafetensorsDtype,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Data offsets in file [start, end)
    pub data_offsets: [usize; 2],
}

/// Safetensors model container
#[derive(Debug, Clone)]
pub struct SafetensorsModel {
    /// Tensor metadata
    pub tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Raw tensor data (not parsed yet)
    pub data: Vec<u8>,
}

impl SafetensorsModel {
    /// Parse Safetensors file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Safetensors file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid header
    /// - Malformed JSON metadata
    /// - Invalid data offsets
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.safetensors")?;
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header (8-byte metadata length)
        let metadata_len = Self::parse_header(&mut cursor)?;

        // Parse JSON metadata
        let tensors = Self::parse_metadata(&mut cursor, metadata_len)?;

        // Store remaining data
        let data_start =
            usize::try_from(8 + metadata_len).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_data_offset".to_string(),
                reason: format!(
                    "Data offset {} exceeds platform usize limit",
                    8 + metadata_len
                ),
            })?;
        let data = data[data_start..].to_vec();

        Ok(Self { tensors, data })
    }

    /// Extract F32 tensor data by name
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Tensor dtype is not F32
    /// - Data offsets are invalid
    ///
    /// # Panics
    ///
    /// Never panics. The `unwrap()` in byte conversion is safe because
    /// `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// let weights = model.get_tensor_f32("layer1.weight")?;
    /// println!("Weights: {:?}", weights);
    /// ```
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        // Find tensor metadata
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Verify dtype is F32
        if tensor.dtype != SafetensorsDtype::F32 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected F32"),
            });
        }

        // Extract data slice
        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert bytes to f32 vector
        if !bytes.len().is_multiple_of(4) {
            let len = bytes.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data size {len} is not a multiple of 4"),
            });
        }

        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();

        Ok(values)
    }

    /// Parse header (8-byte metadata length)
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_len".to_string(),
                reason: e.to_string(),
            })?;

        Ok(u64::from_le_bytes(buf))
    }

    /// Parse JSON metadata
    fn parse_metadata(
        cursor: &mut Cursor<&[u8]>,
        len: u64,
    ) -> Result<HashMap<String, SafetensorsTensorInfo>> {
        // Read JSON bytes
        let len_usize = usize::try_from(len).map_err(|_| RealizarError::UnsupportedOperation {
            operation: "convert_metadata_len".to_string(),
            reason: format!("Metadata length {len} exceeds platform usize limit"),
        })?;

        let mut json_bytes = vec![0u8; len_usize];
        cursor
            .read_exact(&mut json_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_json".to_string(),
                reason: e.to_string(),
            })?;

        // Parse JSON as generic Value first to handle __metadata__ and other special keys
        let json_value: serde_json::Value = serde_json::from_slice(&json_bytes).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "parse_json".to_string(),
                reason: e.to_string(),
            }
        })?;

        let json_map =
            json_value
                .as_object()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "parse_json".to_string(),
                    reason: "Expected JSON object".to_string(),
                })?;

        // Convert to SafetensorsTensorInfo, skipping special keys like __metadata__
        let mut tensors = HashMap::new();
        for (name, value) in json_map {
            // Skip metadata keys (start with __)
            if name.starts_with("__") {
                continue;
            }

            // Parse tensor metadata
            let meta: TensorMetadata = serde_json::from_value(value.clone()).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "parse_tensor_metadata".to_string(),
                    reason: format!("Failed to parse tensor '{name}': {e}"),
                }
            })?;

            tensors.insert(
                name.clone(),
                SafetensorsTensorInfo {
                    name: name.clone(),
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offsets: meta.data_offsets,
                },
            );
        }

        Ok(tensors)
    }

    /// Get tensor data as F16 values (converts to F32)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F16
    pub fn get_tensor_f16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F16 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected F16"),
            });
        }

        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert F16 bytes to F32
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        Ok(values)
    }

    /// Get tensor data as BF16 values (converts to F32)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16
    pub fn get_tensor_bf16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::BF16 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected BF16"),
            });
        }

        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert BF16 bytes to F32 using SIMD-accelerated conversion
        // This provides 3-4x speedup over scalar conversion
        let values = simd_bf16_to_f32(bytes);

        Ok(values)
    }

    /// Get tensor as F32 with automatic dtype conversion
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not supported
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        match tensor.dtype {
            SafetensorsDtype::F32 => self.get_tensor_f32(name),
            SafetensorsDtype::F16 => self.get_tensor_f16_as_f32(name),
            SafetensorsDtype::BF16 => self.get_tensor_bf16_as_f32(name),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Unsupported dtype {:?} for tensor '{name}'", tensor.dtype),
            }),
        }
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.tensors.get(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

// ============================================================================
// SafeTensors Config loader (for sibling config.json)
// ============================================================================

/// Model configuration from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsConfig {
    /// Hidden dimension
    #[serde(alias = "n_embd", alias = "d_model")]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    #[serde(alias = "n_layer", alias = "num_layers")]
    pub num_hidden_layers: Option<usize>,
    /// Number of attention heads
    #[serde(alias = "n_head")]
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Intermediate/FFN dimension
    #[serde(alias = "n_inner")]
    pub intermediate_size: Option<usize>,
    /// Maximum sequence length
    #[serde(alias = "n_positions", alias = "n_ctx")]
    pub max_position_embeddings: Option<usize>,
    /// RMSNorm epsilon
    pub rms_norm_eps: Option<f32>,
    /// RoPE theta
    pub rope_theta: Option<f32>,
    /// Model architecture name
    pub architectures: Option<Vec<String>>,
    /// Model type
    pub model_type: Option<String>,
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
}

impl SafetensorsConfig {
    /// Load config from sibling config.json file
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (config.json will be loaded from same directory)
    ///
    /// # Returns
    ///
    /// Config if found and parsed, None otherwise
    pub fn load_from_sibling(model_path: &std::path::Path) -> Option<Self> {
        let config_path = model_path.with_file_name("config.json");
        if !config_path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&config_path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Get number of key-value heads (defaults to num_attention_heads for MHA)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .or(self.num_attention_heads)
            .unwrap_or(1)
    }

    /// Get model architecture string
    #[must_use]
    pub fn architecture(&self) -> String {
        self.architectures
            .as_ref()
            .and_then(|a| a.first())
            .cloned()
            .or_else(|| self.model_type.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

// ============================================================================
// Zero-Copy Memory-Mapped SafeTensors Model (T-QA-020)
// ============================================================================

/// Zero-copy memory-mapped SafeTensors model container
///
/// Unlike `SafetensorsModel` which copies all tensor data to the heap,
/// `MappedSafeTensorsModel` uses memory-mapping (mmap) for true zero-copy
/// access to tensor data. This is critical for fast model loading (TTFT).
///
/// # Performance Characteristics
///
/// - **Loading time**: O(1) - only parses header/metadata, no data copy
/// - **Memory**: Only RSS grows as pages are accessed (demand paging)
/// - **TTFT target**: < 500ms for 3GB model
///
/// # Example
///
/// ```rust,ignore
/// let model = MappedSafeTensorsModel::load("model.safetensors")?;
/// let weights = model.get_tensor_bytes("layer1.weight")?;
/// // weights is a zero-copy slice into the mmap'd file
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct MappedSafeTensorsModel {
    /// Memory-mapped file data
    mmap: memmap2::Mmap,
    /// File path (for diagnostics)
    path: std::path::PathBuf,
    /// Tensor metadata (parsed from header)
    tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Offset where tensor data begins (after header + JSON metadata)
    data_offset: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl MappedSafeTensorsModel {
    /// Load a SafeTensors file with zero-copy memory mapping
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SafeTensors file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be opened
    /// - Memory mapping fails
    /// - Header/metadata parsing fails
    ///
    /// # Performance
    ///
    /// This method is O(1) with respect to file size - only the header
    /// and JSON metadata are parsed. Tensor data is not touched until
    /// `get_tensor_bytes()` is called.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open file
        let file = std::fs::File::open(&path).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "open_safetensors".to_string(),
            reason: format!("Failed to open file '{}': {}", path.display(), e),
        })?;

        // Memory-map the file (zero-copy)
        // SAFETY: File is opened read-only and we don't modify it
        let mmap = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "mmap_safetensors".to_string(),
                    reason: format!("Failed to mmap file '{}': {}", path.display(), e),
                }
            })?
        };

        // Parse header (8-byte metadata length)
        if mmap.len() < 8 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!(
                    "File too small: {} bytes (minimum 8 for header)",
                    mmap.len()
                ),
            });
        }

        let metadata_len =
            u64::from_le_bytes(mmap[0..8].try_into().expect("slice is exactly 8 bytes"));

        let metadata_len_usize =
            usize::try_from(metadata_len).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!("Metadata length {} exceeds platform limit", metadata_len),
            })?;

        // Verify we have enough data for metadata
        let data_offset = 8 + metadata_len_usize;
        if mmap.len() < data_offset {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!(
                    "File truncated: need {} bytes for header+metadata, have {}",
                    data_offset,
                    mmap.len()
                ),
            });
        }

        // Parse JSON metadata (from mmap'd memory, no copy)
        let json_bytes = &mmap[8..data_offset];
        let tensors = Self::parse_metadata(json_bytes)?;

        Ok(Self {
            mmap,
            path,
            tensors,
            data_offset,
        })
    }

    /// Parse JSON metadata from bytes
    fn parse_metadata(json_bytes: &[u8]) -> Result<HashMap<String, SafetensorsTensorInfo>> {
        // Parse JSON as generic Value first to handle __metadata__ and other special keys
        let json_value: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "parse_json".to_string(),
                reason: e.to_string(),
            }
        })?;

        let json_map =
            json_value
                .as_object()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "parse_json".to_string(),
                    reason: "Expected JSON object".to_string(),
                })?;

        // Convert to SafetensorsTensorInfo, skipping special keys like __metadata__
        let mut tensors = HashMap::new();
        for (name, value) in json_map {
            // Skip metadata keys (start with __)
            if name.starts_with("__") {
                continue;
            }

            // Parse tensor metadata
            let meta: TensorMetadata = serde_json::from_value(value.clone()).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "parse_tensor_metadata".to_string(),
                    reason: format!("Failed to parse tensor '{name}': {e}"),
                }
            })?;

            tensors.insert(
                name.clone(),
                SafetensorsTensorInfo {
                    name: name.clone(),
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offsets: meta.data_offsets,
                },
            );
        }

        Ok(tensors)
    }

    /// Get raw tensor bytes (zero-copy slice into mmap'd file)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Returns
    ///
    /// Zero-copy slice into the memory-mapped file. The slice is valid
    /// as long as `self` is alive.
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or offsets are invalid.
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        let [start, end] = tensor.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > self.mmap.len() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' data offsets [{}, {}] exceed file size {}",
                    name,
                    abs_start,
                    abs_end,
                    self.mmap.len()
                ),
            });
        }

        Ok(&self.mmap[abs_start..abs_end])
    }

    /// Get tensor as F32 values (zero-copy bytes, then convert)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F32.
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F32 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected F32",
                    name, tensor.dtype
                ),
            });
        }

        let bytes = self.get_tensor_bytes(name)?;

        if !bytes.len().is_multiple_of(4) {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data size {} is not a multiple of 4", bytes.len()),
            });
        }

        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();

        Ok(values)
    }

    /// Get tensor as BF16 bytes (zero-copy, native format)
    ///
    /// Returns raw BF16 bytes for native SIMD processing without
    /// F32 conversion at boot time.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16.
    pub fn get_tensor_bf16_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::BF16 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected BF16",
                    name, tensor.dtype
                ),
            });
        }

        self.get_tensor_bytes(name)
    }

    /// Get tensor as BF16 values converted to F32
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16.
    pub fn get_tensor_bf16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_bf16_bytes(name)?;

        // Convert BF16 bytes to F32 using SIMD-accelerated conversion
        // This provides 3-4x speedup over scalar conversion
        let values = simd_bf16_to_f32(bytes);

        Ok(values)
    }

    /// Get tensor as F16 bytes (zero-copy, native format)
    ///
    /// Returns raw F16 bytes for native SIMD processing.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F16.
    pub fn get_tensor_f16_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F16 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected F16",
                    name, tensor.dtype
                ),
            });
        }

        self.get_tensor_bytes(name)
    }

    /// Get tensor as F16 values converted to F32
    pub fn get_tensor_f16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_f16_bytes(name)?;

        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        Ok(values)
    }

    /// Get tensor as F32 with automatic dtype conversion
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        match tensor.dtype {
            SafetensorsDtype::F32 => self.get_tensor_f32(name),
            SafetensorsDtype::F16 => self.get_tensor_f16_as_f32(name),
            SafetensorsDtype::BF16 => self.get_tensor_bf16_as_f32(name),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Unsupported dtype {:?} for tensor '{}'", tensor.dtype, name),
            }),
        }
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.tensors.get(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get the file path
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Get the total file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get the number of tensors
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_safetensors() {
        // Minimal valid Safetensors: 8-byte header + empty JSON "{}"
        let mut data = Vec::new();
        data.extend_from_slice(&2u64.to_le_bytes()); // metadata_len = 2
        data.extend_from_slice(b"{}"); // Empty JSON

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 0);
        assert_eq!(model.data.len(), 0);
    }

    #[test]
    fn test_invalid_header_truncated() {
        // Only 4 bytes (should be 8)
        let data = [0u8; 4];
        let result = SafetensorsModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let data = &[];
        let result = SafetensorsModel::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_single_tensor() {
        // Safetensors with one F32 tensor
        let json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        // Add 24 bytes of dummy tensor data (2*3*4 = 24 bytes for F32)
        data.extend_from_slice(&[0u8; 24]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 1);

        let tensor = model.tensors.get("weight").expect("test");
        assert_eq!(tensor.name, "weight");
        assert_eq!(tensor.dtype, SafetensorsDtype::F32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data_offsets, [0, 24]);
    }

    #[test]
    fn test_parse_multiple_tensors() {
        // Safetensors with multiple tensors of different types
        let json = r#"{
            "layer1.weight":{"dtype":"F32","shape":[128,256],"data_offsets":[0,131072]},
            "layer1.bias":{"dtype":"F32","shape":[128],"data_offsets":[131072,131584]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        // Add dummy tensor data
        data.extend_from_slice(&vec![0u8; 131_584]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 2);

        let weight = model.tensors.get("layer1.weight").expect("test");
        assert_eq!(weight.dtype, SafetensorsDtype::F32);
        assert_eq!(weight.shape, vec![128, 256]);
        assert_eq!(weight.data_offsets, [0, 131_072]);

        let bias = model.tensors.get("layer1.bias").expect("test");
        assert_eq!(bias.dtype, SafetensorsDtype::F32);
        assert_eq!(bias.shape, vec![128]);
        assert_eq!(bias.data_offsets, [131_072, 131_584]);
    }

    #[test]
    fn test_parse_various_dtypes() {
        // Test different data types
        let json = r#"{
            "f32_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "i32_tensor":{"dtype":"I32","shape":[2],"data_offsets":[8,16]},
            "u8_tensor":{"dtype":"U8","shape":[4],"data_offsets":[16,20]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 20]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 3);

        assert_eq!(
            model.tensors.get("f32_tensor").expect("test").dtype,
            SafetensorsDtype::F32
        );
        assert_eq!(
            model.tensors.get("i32_tensor").expect("test").dtype,
            SafetensorsDtype::I32
        );
        assert_eq!(
            model.tensors.get("u8_tensor").expect("test").dtype,
            SafetensorsDtype::U8
        );
    }

    #[test]
    fn test_invalid_json_error() {
        // Invalid JSON in metadata
        let mut data = Vec::new();
        data.extend_from_slice(&10u64.to_le_bytes()); // metadata_len = 10
        data.extend_from_slice(b"not json!!"); // Invalid JSON

        let result = SafetensorsModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::UnsupportedOperation { .. }
        ));
    }

    #[test]
    fn test_truncated_json_error() {
        // Header says JSON is longer than actual data
        let mut data = Vec::new();
        data.extend_from_slice(&100u64.to_le_bytes()); // metadata_len = 100
        data.extend_from_slice(b"{}"); // Only 2 bytes, not 100

        let result = SafetensorsModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::UnsupportedOperation { .. }
        ));
    }

    #[test]
    fn test_parse_all_dtypes() {
        // Test all supported data types
        let json = r#"{
            "f32":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "f16":{"dtype":"F16","shape":[1],"data_offsets":[4,6]},
            "bf16":{"dtype":"BF16","shape":[1],"data_offsets":[6,8]},
            "i32":{"dtype":"I32","shape":[1],"data_offsets":[8,12]},
            "i64":{"dtype":"I64","shape":[1],"data_offsets":[12,20]},
            "u8":{"dtype":"U8","shape":[1],"data_offsets":[20,21]},
            "bool":{"dtype":"Bool","shape":[1],"data_offsets":[21,22]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 22]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 7);

        assert_eq!(
            model.tensors.get("f32").expect("test").dtype,
            SafetensorsDtype::F32
        );
        assert_eq!(
            model.tensors.get("f16").expect("test").dtype,
            SafetensorsDtype::F16
        );
        assert_eq!(
            model.tensors.get("bf16").expect("test").dtype,
            SafetensorsDtype::BF16
        );
        assert_eq!(
            model.tensors.get("i32").expect("test").dtype,
            SafetensorsDtype::I32
        );
        assert_eq!(
            model.tensors.get("i64").expect("test").dtype,
            SafetensorsDtype::I64
        );
        assert_eq!(
            model.tensors.get("u8").expect("test").dtype,
            SafetensorsDtype::U8
        );
        assert_eq!(
            model.tensors.get("bool").expect("test").dtype,
            SafetensorsDtype::Bool
        );
    }

    #[test]
    fn test_tensor_data_preserved() {
        // Verify tensor data is correctly preserved
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        // Add specific tensor data (two f32s: 1.0 and 2.0)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.data.len(), 8);

        // Verify we can read back the f32 values
        let val1 = f32::from_le_bytes(model.data[0..4].try_into().expect("test"));
        let val2 = f32::from_le_bytes(model.data[4..8].try_into().expect("test"));
        assert!((val1 - 1.0).abs() < 1e-6);
        assert!((val2 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_multidimensional_shapes() {
        // Test tensors with various shapes
        let json = r#"{
            "scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
            "vector":{"dtype":"F32","shape":[10],"data_offsets":[4,44]},
            "matrix":{"dtype":"F32","shape":[3,4],"data_offsets":[44,92]},
            "tensor3d":{"dtype":"F32","shape":[2,3,4],"data_offsets":[92,188]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 188]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 4);

        assert_eq!(
            model.tensors.get("scalar").expect("test").shape,
            Vec::<usize>::new()
        );
        assert_eq!(model.tensors.get("vector").expect("test").shape, vec![10]);
        assert_eq!(model.tensors.get("matrix").expect("test").shape, vec![3, 4]);
        assert_eq!(
            model.tensors.get("tensor3d").expect("test").shape,
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_aprender_linear_regression_format_compatibility() {
        // Test aprender LinearRegression SafeTensors format compatibility
        // Format: {"coefficients": [n_features], "intercept": [1]}
        // Example model: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5

        let json = r#"{
            "coefficients":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "intercept":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());

        // Metadata
        data.extend_from_slice(json_bytes);

        // Tensor data: coefficients [2.0, 3.0, 1.5]
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&1.5f32.to_le_bytes());

        // intercept [0.5]
        data.extend_from_slice(&0.5f32.to_le_bytes());

        // Parse with realizar
        let model = SafetensorsModel::from_bytes(&data).expect("test");

        // Verify structure
        assert_eq!(model.tensors.len(), 2);

        // Check coefficients tensor
        let coef = model.tensors.get("coefficients").expect("test");
        assert_eq!(coef.dtype, SafetensorsDtype::F32);
        assert_eq!(coef.shape, vec![3]);
        assert_eq!(coef.data_offsets, [0, 12]);

        // Check intercept tensor
        let intercept = model.tensors.get("intercept").expect("test");
        assert_eq!(intercept.dtype, SafetensorsDtype::F32);
        assert_eq!(intercept.shape, vec![1]);
        assert_eq!(intercept.data_offsets, [12, 16]);

        // Verify we can extract the actual values
        let coef_vals: Vec<f32> = (0..3)
            .map(|i| {
                let offset = i * 4;
                f32::from_le_bytes(model.data[offset..offset + 4].try_into().expect("test"))
            })
            .collect();
        assert!((coef_vals[0] - 2.0).abs() < 1e-6);
        assert!((coef_vals[1] - 3.0).abs() < 1e-6);
        assert!((coef_vals[2] - 1.5).abs() < 1e-6);

        let intercept_val = f32::from_le_bytes(model.data[12..16].try_into().expect("test"));
        assert!((intercept_val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_tensor_f32_helper() {
        // Test the get_tensor_f32 helper method
        let json = r#"{
            "weights":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
            "bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);

        // weights: [1.0, 2.0, 3.0, 4.0]
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());

        // bias: [0.5, 0.25]
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.25f32.to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");

        // Test extracting weights
        let weights = model.get_tensor_f32("weights").expect("test");
        assert_eq!(weights.len(), 4);
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 2.0).abs() < 1e-6);
        assert!((weights[2] - 3.0).abs() < 1e-6);
        assert!((weights[3] - 4.0).abs() < 1e-6);

        // Test extracting bias
        let bias = model.get_tensor_f32("bias").expect("test");
        assert_eq!(bias.len(), 2);
        assert!((bias[0] - 0.5).abs() < 1e-6);
        assert!((bias[1] - 0.25).abs() < 1e-6);

        // Test error: tensor not found
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_wrong_dtype() {
        // Test error when tensor has wrong dtype
        let json = r#"{
            "int_tensor":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&2i32.to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");

        // Should error because dtype is I32, not F32
        let result = model.get_tensor_f32("int_tensor");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_with_aprender_model() {
        // Use get_tensor_f32 with aprender LinearRegression format
        let json = r#"{
            "coefficients":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "intercept":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&1.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");

        // Extract using helper method - much cleaner!
        let coefficients = model.get_tensor_f32("coefficients").expect("test");
        assert_eq!(coefficients, vec![2.0, 3.0, 1.5]);

        let intercept = model.get_tensor_f32("intercept").expect("test");
        assert_eq!(intercept, vec![0.5]);
    }

    // ========== Coverage tests for untested functions ==========

    #[test]
    fn test_cov_get_tensor_f16_as_f32() {
        let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        // Two F16 values: 1.0 and 2.0
        data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let weights = model.get_tensor_f16_as_f32("weights").expect("test");

        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 0.01);
        assert!((weights[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_cov_get_tensor_f16_not_found() {
        let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 4]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_f16_as_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_f16_wrong_dtype() {
        let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_f16_as_f32("weights");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_f16_data_offset_exceeds() {
        let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,100]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 4]); // Only 4 bytes, offset says 100

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_f16_as_f32("weights");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_bf16_as_f32() {
        let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        // Two BF16 values: 1.0 and 2.0
        data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let weights = model.get_tensor_bf16_as_f32("weights").expect("test");

        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 0.01);
        assert!((weights[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_cov_get_tensor_bf16_not_found() {
        let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 4]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_bf16_as_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_bf16_wrong_dtype() {
        let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_bf16_as_f32("weights");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_bf16_data_offset_exceeds() {
        let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,100]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 4]); // Only 4 bytes

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_bf16_as_f32("weights");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_auto_f32() {
        let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let weights = model.get_tensor_auto("weights").expect("test");
        assert_eq!(weights, vec![1.0, 2.0]);
    }

    #[test]
    fn test_cov_get_tensor_auto_f16() {
        let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let weights = model.get_tensor_auto("weights").expect("test");
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_cov_get_tensor_auto_bf16() {
        let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let weights = model.get_tensor_auto("weights").expect("test");
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_cov_get_tensor_auto_unsupported_dtype() {
        let json = r#"{"weights":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_auto("weights");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_auto_not_found() {
        let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_auto("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_tensor_names() {
        let json = r#"{
            "weight1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "weight2":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 16]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let names = model.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight1"));
        assert!(names.contains(&"weight2"));
    }

    #[test]
    fn test_cov_get_tensor_info() {
        let json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 24]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");

        let info = model.get_tensor_info("weight");
        assert!(info.is_some());
        let info = info.expect("operation failed");
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.dtype, SafetensorsDtype::F32);

        assert!(model.get_tensor_info("nonexistent").is_none());
    }

    #[test]
    fn test_cov_has_tensor() {
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("nonexistent"));
    }

    #[test]
    fn test_cov_get_tensor_f32_data_offset_exceeds() {
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,100]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]); // Only 8 bytes, offset says 100

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_get_tensor_f32_not_multiple_of_4() {
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 7]); // 7 bytes, not a multiple of 4

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_safetensors_config_num_kv_heads() {
        let config = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: Some(4),
            vocab_size: Some(32000),
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.num_kv_heads(), 4);
    }

    #[test]
    fn test_cov_safetensors_config_num_kv_heads_default() {
        let config = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None, // Not set, should fall back to attention heads
            vocab_size: Some(32000),
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.num_kv_heads(), 12);
    }

    #[test]
    fn test_cov_safetensors_config_num_kv_heads_fallback() {
        let config = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: Some(32000),
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.num_kv_heads(), 1); // Fallback to 1
    }

    #[test]
    fn test_cov_safetensors_config_architecture_from_architectures() {
        let config = SafetensorsConfig {
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            model_type: Some("llama".to_string()),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.architecture(), "LlamaForCausalLM");
    }

    #[test]
    fn test_cov_safetensors_config_architecture_from_model_type() {
        let config = SafetensorsConfig {
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: None,
            model_type: Some("llama".to_string()),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.architecture(), "llama");
    }

    #[test]
    fn test_cov_safetensors_config_architecture_unknown() {
        let config = SafetensorsConfig {
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.architecture(), "unknown");
    }

    #[test]
    fn test_cov_safetensors_config_load_from_sibling_not_found() {
        let path = std::path::Path::new("/nonexistent/model.safetensors");
        let config = SafetensorsConfig::load_from_sibling(path);
        assert!(config.is_none());
    }

    #[test]
    fn test_cov_metadata_key_skipped() {
        // Test that __metadata__ key is skipped
        let json = r#"{
            "__metadata__":{"format":"pt"},
            "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
        }"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let model = SafetensorsModel::from_bytes(&data).expect("test");
        assert_eq!(model.tensors.len(), 1);
        assert!(model.tensors.contains_key("weight"));
        assert!(!model.tensors.contains_key("__metadata__"));
    }

    #[test]
    fn test_cov_json_not_object() {
        // JSON is an array, not an object
        let json = r"[]";
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);

        let result = SafetensorsModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_cov_tensor_metadata_parse_error() {
        // Tensor has invalid metadata (missing dtype)
        let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(json_bytes);
        data.extend_from_slice(&[0u8; 8]);

        let result = SafetensorsModel::from_bytes(&data);
        assert!(result.is_err());
    }

    // ========== MappedSafeTensorsModel tests ==========

    #[cfg(not(target_arch = "wasm32"))]
    mod mapped_tests {
        use super::*;
        use std::io::Write;

        /// Helper to create a temporary safetensors file
        fn create_temp_safetensors(
            tensors: &[(&str, SafetensorsDtype, &[usize], &[u8])],
        ) -> tempfile::NamedTempFile {
            let mut json_map = serde_json::Map::new();
            let mut tensor_data = Vec::new();
            let mut offset = 0usize;

            for (name, dtype, shape, data) in tensors {
                let dtype_str = match dtype {
                    SafetensorsDtype::F32 => "F32",
                    SafetensorsDtype::F16 => "F16",
                    SafetensorsDtype::BF16 => "BF16",
                    SafetensorsDtype::I32 => "I32",
                    SafetensorsDtype::I64 => "I64",
                    SafetensorsDtype::U8 => "U8",
                    SafetensorsDtype::Bool => "Bool",
                };

                let end = offset + data.len();
                json_map.insert(
                    (*name).to_string(),
                    serde_json::json!({
                        "dtype": dtype_str,
                        "shape": shape,
                        "data_offsets": [offset, end]
                    }),
                );

                tensor_data.extend_from_slice(data);
                offset = end;
            }

            let json_str = serde_json::to_string(&json_map).expect("JSON serialization");
            let json_bytes = json_str.as_bytes();

            let mut file = tempfile::NamedTempFile::new().expect("temp file creation");
            file.write_all(&(json_bytes.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json_bytes).expect("write metadata");
            file.write_all(&tensor_data).expect("write tensor data");
            file.flush().expect("flush file");

            file
        }

        #[test]
        fn test_mapped_load_basic() {
            // Create temp file with one F32 tensor
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.tensor_count(), 1);
            assert!(model.has_tensor("weight"));
            assert!(!model.has_tensor("nonexistent"));
        }

        #[test]
        fn test_mapped_file_not_found() {
            let result = MappedSafeTensorsModel::load("/nonexistent/path/model.safetensors");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_file_too_small() {
            // Create a file with only 4 bytes (less than header size)
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            file.write_all(&[0u8; 4]).expect("write");
            file.flush().expect("flush");

            let result = MappedSafeTensorsModel::load(file.path());
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("File too small"),
                "Expected 'File too small' error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_truncated_metadata() {
            // Create a file that claims metadata is 100 bytes but only has 10
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            file.write_all(&100u64.to_le_bytes()).expect("write header");
            file.write_all(b"{}").expect("write short json");
            file.flush().expect("flush");

            let result = MappedSafeTensorsModel::load(file.path());
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("truncated"),
                "Expected 'truncated' error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_invalid_json() {
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let invalid_json = b"not valid json!!";
            file.write_all(&(invalid_json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(invalid_json).expect("write json");
            file.flush().expect("flush");

            let result = MappedSafeTensorsModel::load(file.path());
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_json_not_object() {
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let json = b"[]"; // Array instead of object
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json).expect("write json");
            file.flush().expect("flush");

            let result = MappedSafeTensorsModel::load(file.path());
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("Expected JSON object"),
                "Expected 'Expected JSON object' error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_tensor_metadata_parse_error() {
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            // Missing dtype field
            let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json.as_bytes()).expect("write json");
            file.write_all(&[0u8; 8]).expect("write data");
            file.flush().expect("flush");

            let result = MappedSafeTensorsModel::load(file.path());
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("Failed to parse tensor"),
                "Expected tensor parse error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_bytes() {
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let bytes = model.get_tensor_bytes("weight").expect("get bytes");
            assert_eq!(bytes.len(), 12); // 3 * 4 bytes
        }

        #[test]
        fn test_mapped_get_tensor_bytes_not_found() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &0.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_bytes("nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_get_tensor_bytes_offset_exceeds() {
            // Create a file with a tensor that claims more data than exists
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let json = r#"{"weight":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json.as_bytes()).expect("write json");
            file.write_all(&[0u8; 8])
                .expect("write only 8 bytes of data");
            file.flush().expect("flush");

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_bytes("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("exceed file size"),
                "Expected 'exceed file size' error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_f32() {
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[4], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model.get_tensor_f32("weight").expect("get f32");
            assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_mapped_get_tensor_f32_not_found() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_f32("nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_get_tensor_f32_wrong_dtype() {
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_f32("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("expected F32"),
                "Expected wrong dtype error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_f32_not_multiple_of_4() {
            // Create file with misaligned data
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let json = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,7]}}"#;
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json.as_bytes()).expect("write json");
            file.write_all(&[0u8; 7]).expect("write 7 bytes");
            file.flush().expect("flush");

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_f32("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("not a multiple of 4"),
                "Expected alignment error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_f16_bytes() {
            let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let bytes = model.get_tensor_f16_bytes("weight").expect("get f16 bytes");
            assert_eq!(bytes.len(), 4); // 2 * 2 bytes
        }

        #[test]
        fn test_mapped_get_tensor_f16_bytes_not_found() {
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[1], &[0u8; 2])]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_f16_bytes("nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_get_tensor_f16_bytes_wrong_dtype() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_f16_bytes("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("expected F16"),
                "Expected wrong dtype error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_f16_as_f32() {
            let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model
                .get_tensor_f16_as_f32("weight")
                .expect("get f16 as f32");
            assert_eq!(values.len(), 2);
            assert!((values[0] - 1.0).abs() < 0.01);
            assert!((values[1] - 2.0).abs() < 0.01);
        }

        #[test]
        fn test_mapped_get_tensor_bf16_bytes() {
            let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let bytes = model
                .get_tensor_bf16_bytes("weight")
                .expect("get bf16 bytes");
            assert_eq!(bytes.len(), 4); // 2 * 2 bytes
        }

        #[test]
        fn test_mapped_get_tensor_bf16_bytes_not_found() {
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[1], &[0u8; 2])]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_bf16_bytes("nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_get_tensor_bf16_bytes_wrong_dtype() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_bf16_bytes("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("expected BF16"),
                "Expected wrong dtype error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_bf16_as_f32() {
            let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model
                .get_tensor_bf16_as_f32("weight")
                .expect("get bf16 as f32");
            assert_eq!(values.len(), 2);
            assert!((values[0] - 1.0).abs() < 0.01);
            assert!((values[1] - 2.0).abs() < 0.01);
        }

        #[test]
        fn test_mapped_get_tensor_auto_f32() {
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model.get_tensor_auto("weight").expect("get auto");
            assert_eq!(values, vec![1.0, 2.0]);
        }

        #[test]
        fn test_mapped_get_tensor_auto_f16() {
            let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model.get_tensor_auto("weight").expect("get auto");
            assert_eq!(values.len(), 2);
        }

        #[test]
        fn test_mapped_get_tensor_auto_bf16() {
            let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let values = model.get_tensor_auto("weight").expect("get auto");
            assert_eq!(values.len(), 2);
        }

        #[test]
        fn test_mapped_get_tensor_auto_unsupported() {
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_auto("weight");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{err:?}").contains("Unsupported dtype"),
                "Expected unsupported dtype error, got: {err:?}"
            );
        }

        #[test]
        fn test_mapped_get_tensor_auto_not_found() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let result = model.get_tensor_auto("nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_mapped_tensor_names() {
            let file = create_temp_safetensors(&[
                (
                    "weight1",
                    SafetensorsDtype::F32,
                    &[1],
                    &1.0f32.to_le_bytes(),
                ),
                (
                    "weight2",
                    SafetensorsDtype::F32,
                    &[1],
                    &2.0f32.to_le_bytes(),
                ),
            ]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let names = model.tensor_names();
            assert_eq!(names.len(), 2);
            assert!(names.contains(&"weight1"));
            assert!(names.contains(&"weight2"));
        }

        #[test]
        fn test_mapped_get_tensor_info() {
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            let info = model.get_tensor_info("weight");
            assert!(info.is_some());
            let info = info.expect("tensor info");
            assert_eq!(info.dtype, SafetensorsDtype::F32);
            assert_eq!(info.shape, vec![3]);

            assert!(model.get_tensor_info("nonexistent").is_none());
        }

        #[test]
        fn test_mapped_path() {
            let file = create_temp_safetensors(&[(
                "weight",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            )]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.path(), file.path());
        }

        #[test]
        fn test_mapped_file_size() {
            let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let file =
                create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            // File size = 8 (header) + json length + 8 (tensor data)
            assert!(model.file_size() > 8);
        }

        #[test]
        fn test_mapped_tensor_count() {
            let file = create_temp_safetensors(&[
                (
                    "weight1",
                    SafetensorsDtype::F32,
                    &[1],
                    &1.0f32.to_le_bytes(),
                ),
                (
                    "weight2",
                    SafetensorsDtype::F32,
                    &[1],
                    &2.0f32.to_le_bytes(),
                ),
                (
                    "weight3",
                    SafetensorsDtype::F32,
                    &[1],
                    &3.0f32.to_le_bytes(),
                ),
            ]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.tensor_count(), 3);
        }

        #[test]
        fn test_mapped_metadata_key_skipped() {
            // Test that __metadata__ key is skipped in mapped model
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let json = r#"{
                "__metadata__":{"format":"pt"},
                "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
            }"#;
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json.as_bytes()).expect("write json");
            file.write_all(&[0u8; 8]).expect("write data");
            file.flush().expect("flush");

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.tensor_count(), 1);
            assert!(model.has_tensor("weight"));
            assert!(!model.has_tensor("__metadata__"));
        }

        #[test]
        fn test_mapped_multiple_tensors() {
            let w1: Vec<u8> = [1.0f32, 2.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let w2: Vec<u8> = [half::f16::from_f32(3.0), half::f16::from_f32(4.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let w3: Vec<u8> = [half::bf16::from_f32(5.0)]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();

            let file = create_temp_safetensors(&[
                ("f32_tensor", SafetensorsDtype::F32, &[2], &w1),
                ("f16_tensor", SafetensorsDtype::F16, &[2], &w2),
                ("bf16_tensor", SafetensorsDtype::BF16, &[1], &w3),
            ]);

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.tensor_count(), 3);

            let f32_vals = model.get_tensor_f32("f32_tensor").expect("f32");
            assert_eq!(f32_vals, vec![1.0, 2.0]);

            let f16_vals = model.get_tensor_f16_as_f32("f16_tensor").expect("f16");
            assert!((f16_vals[0] - 3.0).abs() < 0.01);
            assert!((f16_vals[1] - 4.0).abs() < 0.01);

            let bf16_vals = model.get_tensor_bf16_as_f32("bf16_tensor").expect("bf16");
            assert!((bf16_vals[0] - 5.0).abs() < 0.01);
        }

        #[test]
        fn test_mapped_empty_tensors() {
            // Test with empty tensor list (only {} metadata)
            let mut file = tempfile::NamedTempFile::new().expect("temp file");
            let json = b"{}";
            file.write_all(&(json.len() as u64).to_le_bytes())
                .expect("write header");
            file.write_all(json).expect("write json");
            file.flush().expect("flush");

            let model = MappedSafeTensorsModel::load(file.path()).expect("load");
            assert_eq!(model.tensor_count(), 0);
            assert!(model.tensor_names().is_empty());
        }
    }

    // ========== Additional SafetensorsConfig coverage ==========

    #[test]
    fn test_cov_safetensors_config_load_from_sibling_invalid_json() {
        // Create a temp dir with an invalid config.json
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, "not valid json").expect("write config");

        let model_path = temp_dir.path().join("model.safetensors");
        let config = SafetensorsConfig::load_from_sibling(&model_path);
        assert!(config.is_none());
    }

    #[test]
    fn test_cov_safetensors_config_load_from_sibling_valid() {
        // Create a temp dir with valid config.json
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let config_path = temp_dir.path().join("config.json");
        let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 32000,
            "model_type": "llama"
        }"#;
        std::fs::write(&config_path, config_json).expect("write config");

        let model_path = temp_dir.path().join("model.safetensors");
        let config = SafetensorsConfig::load_from_sibling(&model_path);
        assert!(config.is_some());
        let config = config.expect("config");
        assert_eq!(config.hidden_size, Some(768));
        assert_eq!(config.num_hidden_layers, Some(12));
        assert_eq!(config.num_attention_heads, Some(12));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.model_type, Some("llama".to_string()));
    }

    #[test]
    fn test_cov_safetensors_config_serde_aliases() {
        // Test that serde aliases work (n_embd, n_layer, n_head, etc.)
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let config_path = temp_dir.path().join("config.json");
        let config_json = r#"{
            "n_embd": 512,
            "n_layer": 6,
            "n_head": 8,
            "n_inner": 2048,
            "n_positions": 1024
        }"#;
        std::fs::write(&config_path, config_json).expect("write config");

        let model_path = temp_dir.path().join("model.safetensors");
        let config = SafetensorsConfig::load_from_sibling(&model_path);
        assert!(config.is_some());
        let config = config.expect("config");
        assert_eq!(config.hidden_size, Some(512)); // n_embd alias
        assert_eq!(config.num_hidden_layers, Some(6)); // n_layer alias
        assert_eq!(config.num_attention_heads, Some(8)); // n_head alias
        assert_eq!(config.intermediate_size, Some(2048)); // n_inner alias
        assert_eq!(config.max_position_embeddings, Some(1024)); // n_positions alias
    }

    #[test]
    fn test_cov_safetensors_config_all_fields() {
        // Test config with all optional fields present
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let config_path = temp_dir.path().join("config.json");
        let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;
        std::fs::write(&config_path, config_json).expect("write config");

        let model_path = temp_dir.path().join("model.safetensors");
        let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");

        assert_eq!(config.hidden_size, Some(768));
        assert_eq!(config.num_hidden_layers, Some(12));
        assert_eq!(config.num_attention_heads, Some(12));
        assert_eq!(config.num_key_value_heads, Some(4));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.intermediate_size, Some(3072));
        assert_eq!(config.max_position_embeddings, Some(2048));
        assert!((config.rms_norm_eps.expect("eps") - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta.expect("theta") - 10000.0).abs() < 1e-3);
        assert_eq!(
            config.architectures,
            Some(vec!["LlamaForCausalLM".to_string()])
        );
        assert_eq!(config.model_type, Some("llama".to_string()));
        assert_eq!(config.bos_token_id, Some(1));
        assert_eq!(config.eos_token_id, Some(2));
    }

    #[test]
    fn test_cov_safetensors_config_architecture_empty_list() {
        // Test architecture() with empty architectures list
        let config = SafetensorsConfig {
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            architectures: Some(vec![]), // Empty list
            model_type: Some("gpt2".to_string()),
            bos_token_id: None,
            eos_token_id: None,
        };

        // Should fall back to model_type when architectures is empty
        assert_eq!(config.architecture(), "gpt2");
    }

    // ========== Additional SafetensorsTensorInfo tests ==========

    #[test]
    fn test_cov_tensor_info_clone_and_eq() {
        let info1 = SafetensorsTensorInfo {
            name: "weight".to_string(),
            dtype: SafetensorsDtype::F32,
            shape: vec![2, 3],
            data_offsets: [0, 24],
        };

        let info2 = info1.clone();
        assert_eq!(info1, info2);

        let info3 = SafetensorsTensorInfo {
            name: "weight".to_string(),
            dtype: SafetensorsDtype::F16, // Different dtype
            shape: vec![2, 3],
            data_offsets: [0, 24],
        };
        assert_ne!(info1, info3);
    }

    #[test]
    fn test_cov_safetensors_dtype_clone_and_eq() {
        let dtype1 = SafetensorsDtype::F32;
        let dtype2 = dtype1.clone();
        assert_eq!(dtype1, dtype2);

        let dtype3 = SafetensorsDtype::F16;
        assert_ne!(dtype1, dtype3);
    }
}
