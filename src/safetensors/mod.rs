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
mod tests;
