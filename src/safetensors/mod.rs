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

/// GAP-UX-002: Find sibling companion file with hash-prefix fallback.
///
/// Companion files (config.json, tokenizer.json) may be stored with:
/// 1. Hash prefix: `{stem}.{filename}` (e.g., `d71534cb.config.json`) - PREFERRED
/// 2. Plain name: `{filename}` (e.g., `config.json`) - FALLBACK for backwards compatibility
///
/// # Arguments
///
/// * `model_path` - Path to the model file (e.g., `/cache/d71534cb.safetensors`)
/// * `companion_name` - Name of companion file (e.g., `config.json`)
///
/// # Returns
///
/// Path to the companion file if found, None otherwise
pub fn find_sibling_file(
    model_path: &std::path::Path,
    companion_name: &str,
) -> Option<std::path::PathBuf> {
    let parent = model_path.parent()?;
    let filename = model_path.file_name()?.to_str()?;

    // GH-213: For sharded index.json paths (e.g., "model.safetensors.index.json"),
    // skip the hash-prefix logic and go straight to plain name lookup.
    // For normal model files, use the hash-prefix strategy.
    if !filename.ends_with(".index.json") {
        let stem = model_path.file_stem()?.to_str()?;

        // GAP-UX-002: Try hash-prefixed first (e.g., "d71534cb.config.json")
        let prefixed = parent.join(format!("{stem}.{companion_name}"));
        if prefixed.exists() {
            return Some(prefixed);
        }
    }

    // Fallback: Try plain name for backwards compatibility
    // This is the primary path for sharded models where config.json/tokenizer.json
    // are siblings in the same directory from `apr pull`.
    let plain = parent.join(companion_name);
    if plain.exists() {
        return Some(plain);
    }

    None
}

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
    /// Whether to tie input/output embeddings (lm_head = embed_tokens)
    pub tie_word_embeddings: Option<bool>,
}

impl SafetensorsConfig {
    /// Load config from sibling config.json file
    ///
    /// GAP-UX-002: Tries hash-prefixed companion first (`{stem}.config.json`),
    /// then falls back to non-prefixed (`config.json`) for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (config.json will be loaded from same directory)
    ///
    /// # Returns
    ///
    /// Config if found and parsed, None otherwise
    pub fn load_from_sibling(model_path: &std::path::Path) -> Option<Self> {
        let config_path = find_sibling_file(model_path, "config.json")?;
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

        // GH-213: Validate file covers all tensor data (catches truncated downloads)
        let max_tensor_end = tensors
            .values()
            .map(|t| t.data_offsets[1])
            .max()
            .unwrap_or(0);
        let required_size = data_offset + max_tensor_end;
        if mmap.len() < required_size {
            return Err(RealizarError::UnsupportedOperation {
                operation: "validate_safetensors_size".to_string(),
                reason: format!(
                    "SafeTensors file '{}' is truncated: file has {} bytes but tensor data \
                     requires {} bytes. The file may have been partially downloaded.",
                    path.display(),
                    mmap.len(),
                    required_size
                ),
            });
        }

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

// ============================================================================
// Sharded SafeTensors Model (GH-213)
// ============================================================================

/// JSON structure for model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    /// Mapping from tensor name to shard filename
    weight_map: HashMap<String, String>,
}

/// Sharded SafeTensors model container (GH-213)
///
/// Loads models split across multiple `.safetensors` shard files,
/// as produced by HuggingFace for models >3B parameters.
///
/// # Format
///
/// Sharded models have a `model.safetensors.index.json` that maps
/// tensor names to shard filenames (e.g., `model-00001-of-00002.safetensors`).
///
/// # Example
///
/// ```rust,ignore
/// let model = ShardedSafeTensorsModel::load_from_index("model.safetensors.index.json")?;
/// let weights = model.get_tensor_auto("model.layers.0.self_attn.q_proj.weight")?;
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct ShardedSafeTensorsModel {
    /// Memory-mapped shard files
    shards: Vec<MappedSafeTensorsModel>,
    /// Mapping from tensor name to shard index in `shards`
    tensor_to_shard: HashMap<String, usize>,
    /// Base directory path (parent of index.json)
    base_path: std::path::PathBuf,
    /// Ordered list of unique shard filenames (for deduplication)
    shard_filenames: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ShardedSafeTensorsModel {
    /// Load a sharded SafeTensors model from its index.json file
    ///
    /// Parses the index.json, discovers unique shard files, and mmaps each one.
    ///
    /// # Arguments
    ///
    /// * `index_path` - Path to `model.safetensors.index.json`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - index.json cannot be read or parsed
    /// - Any shard file cannot be opened or mmapped
    pub fn load_from_index(index_path: &std::path::Path) -> Result<Self> {
        let base_path = index_path
            .parent()
            .ok_or_else(|| RealizarError::IoError {
                message: format!(
                    "Cannot determine parent directory of '{}'",
                    index_path.display()
                ),
            })?
            .to_path_buf();

        // Parse index.json
        let index_content =
            std::fs::read_to_string(index_path).map_err(|e| RealizarError::IoError {
                message: format!(
                    "Failed to read index file '{}': {}",
                    index_path.display(),
                    e
                ),
            })?;

        let index: SafetensorsIndex =
            serde_json::from_str(&index_content).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse index.json: {}", e),
            })?;

        // Discover unique shard filenames (preserving order)
        let mut shard_filenames: Vec<String> = Vec::new();
        let mut filename_to_idx: HashMap<String, usize> = HashMap::new();

        for shard_file in index.weight_map.values() {
            if !filename_to_idx.contains_key(shard_file) {
                let idx = shard_filenames.len();
                filename_to_idx.insert(shard_file.clone(), idx);
                shard_filenames.push(shard_file.clone());
            }
        }

        // Load each shard via mmap
        let mut shards = Vec::with_capacity(shard_filenames.len());
        for filename in &shard_filenames {
            let shard_path = base_path.join(filename);
            let shard = MappedSafeTensorsModel::load(&shard_path)?;
            shards.push(shard);
        }

        // Build tensor-to-shard lookup
        let mut tensor_to_shard = HashMap::with_capacity(index.weight_map.len());
        for (tensor_name, shard_file) in &index.weight_map {
            let shard_idx = filename_to_idx[shard_file];
            tensor_to_shard.insert(tensor_name.clone(), shard_idx);
        }

        Ok(Self {
            shards,
            tensor_to_shard,
            base_path,
            shard_filenames,
        })
    }

    /// Get tensor as F32 with automatic dtype conversion (routes to correct shard)
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let shard_idx =
            self.tensor_to_shard
                .get(name)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "get_tensor_auto".to_string(),
                    reason: format!("Tensor '{}' not found in sharded model", name),
                })?;

        self.shards[*shard_idx].get_tensor_auto(name)
    }

    /// Get list of all tensor names across all shards
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_shard.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name (routes to correct shard)
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        let shard_idx = self.tensor_to_shard.get(name)?;
        self.shards[*shard_idx].get_tensor_info(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }

    /// Get the base directory path
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.base_path
    }

    /// Get total number of tensors across all shards
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_to_shard.len()
    }

    /// Get number of shard files
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

// PMAT-234/235: Validation contract - makes bad loads IMPOSSIBLE
// Implements Poka-Yoke (mistake-proofing) via newtype pattern
pub mod validation;
pub use validation::{
    // Runtime validation functions (legacy API)
    enforce_embedding_validation,
    enforce_weight_validation,
    validate_embedding,
    validate_weight,
    // Compile-time enforcement via newtypes (PMAT-235)
    ContractValidationError,
    TensorStats,
    ValidatedAprTransformer,
    ValidatedEmbedding,
    ValidatedVector,
    ValidatedWeight,
    ValidationResult,
};

#[cfg(test)]
mod tests;
