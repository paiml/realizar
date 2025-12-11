//! GGUF (GPT-Generated Unified Format) parser
//!
//! Pure Rust implementation of GGUF binary format reader.
//! Used by llama.cpp, Ollama, and compatible tools.
//!
//! Format specification: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## Format Overview
//!
//! ```text
//! GGUF := HEADER METADATA[] TENSOR_INFO[] TENSOR_DATA[]
//!
//! HEADER := {
//!   magic: u32 = 0x46554747 ("GGUF")
//!   version: u32
//!   tensor_count: u64
//!   metadata_count: u64
//! }
//! ```

use std::{
    collections::HashMap,
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

use memmap2::Mmap;

use crate::error::{RealizarError, Result};

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF versions
pub const GGUF_VERSION_V3: u32 = 3;

/// GGUF quantization type: F32 (unquantized float32)
pub const GGUF_TYPE_F32: u32 = 0;

/// GGUF quantization type: F16 (half precision float16)
pub const GGUF_TYPE_F16: u32 = 1;

/// GGUF quantization type: `Q4_0` (4-bit quantization, block size 32)
pub const GGUF_TYPE_Q4_0: u32 = 2;

/// GGUF quantization type: `Q4_1` (4-bit quantization with min, block size 32)
pub const GGUF_TYPE_Q4_1: u32 = 3;

/// GGUF quantization type: `Q5_0` (5-bit quantization, block size 32)
pub const GGUF_TYPE_Q5_0: u32 = 6;

/// GGUF quantization type: `Q5_1` (5-bit quantization with min, block size 32)
pub const GGUF_TYPE_Q5_1: u32 = 7;

/// GGUF quantization type: `Q8_0` (8-bit quantization, block size 32)
pub const GGUF_TYPE_Q8_0: u32 = 8;

/// GGUF quantization type: `Q4_K` (4-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q4_K: u32 = 12;

/// GGUF quantization type: `Q5_K` (5-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q5_K: u32 = 13;

/// GGUF quantization type: `Q6_K` (6-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q6_K: u32 = 14;

/// GGUF metadata value types
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    /// Unsigned 8-bit integer
    UInt8(u8),
    /// Signed 8-bit integer
    Int8(i8),
    /// Unsigned 16-bit integer
    UInt16(u16),
    /// Signed 16-bit integer
    Int16(i16),
    /// Unsigned 32-bit integer
    UInt32(u32),
    /// Signed 32-bit integer
    Int32(i32),
    /// 32-bit floating point
    Float32(f32),
    /// Boolean
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values
    Array(Vec<GGUFValue>),
    /// Unsigned 64-bit integer
    UInt64(u64),
    /// Signed 64-bit integer
    Int64(i64),
    /// 64-bit floating point
    Float64(f64),
}

/// GGUF file header
#[derive(Debug, Clone, PartialEq)]
pub struct GGUFHeader {
    /// Magic number (must be `GGUF_MAGIC`)
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
}

/// Tensor information
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions (shape)
    pub dims: Vec<u64>,
    /// Quantization type
    pub qtype: u32,
    /// Offset in the file where tensor data starts
    pub offset: u64,
}

/// GGUF alignment requirement (32 bytes)
pub const GGUF_ALIGNMENT: usize = 32;

/// GGUF model container
#[derive(Debug, Clone)]
pub struct GGUFModel {
    /// File header
    pub header: GGUFHeader,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
    /// Offset where tensor data starts (after header/metadata/tensor_info + alignment)
    pub tensor_data_start: usize,
}

/// Memory-mapped GGUF model for zero-copy loading
///
/// Per Dean & Barroso (2013) "The Tail at Scale", memory-mapped I/O eliminates
/// the need to copy file data into process memory, reducing load time and
/// allowing the OS to manage the page cache efficiently.
///
/// # Performance Benefits
///
/// - **Zero-copy loading**: File contents accessed directly via virtual memory
/// - **Lazy loading**: Only pages accessed are read from disk
/// - **Page cache sharing**: Multiple processes can share the same physical pages
/// - **Reduced memory pressure**: Large models don't need to be fully resident
///
/// # Examples
///
/// ```rust,ignore
/// let model = MappedGGUFModel::from_path("model.gguf")?;
/// let tensor_data = model.tensor_data(&tensor_info);
/// ```
pub struct MappedGGUFModel {
    /// Parsed model metadata (header, tensors, etc.)
    pub model: GGUFModel,
    /// Memory-mapped file contents
    mmap: Mmap,
}

impl MappedGGUFModel {
    /// Load GGUF model via memory mapping (zero-copy)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to GGUF model file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be opened
    /// - Memory mapping fails
    /// - GGUF parsing fails (invalid format)
    ///
    /// # Performance
    ///
    /// Memory-mapped loading is faster than `std::fs::read` for large models:
    /// - No file content copy to heap memory
    /// - Kernel handles page management
    /// - Model remains accessible even if larger than RAM (via swap)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = MappedGGUFModel::from_path("phi-2-q4_k_m.gguf")?;
    /// println!("Loaded {} tensors", model.model.tensors.len());
    /// ```
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "open_model_file".to_string(),
            reason: format!("Failed to open {}: {}", path.as_ref().display(), e),
        })?;

        // SAFETY: Memory mapping is safe as long as the file isn't modified
        // while mapped. We only read from the mapping, never write.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "mmap_model_file".to_string(),
                reason: format!("Failed to mmap {}: {}", path.as_ref().display(), e),
            })?
        };

        // Parse the memory-mapped data
        let model = GGUFModel::from_bytes(&mmap)?;

        Ok(Self { model, mmap })
    }

    /// Get the raw memory-mapped file data
    ///
    /// This provides direct access to the file contents without copying.
    /// Use this with tensor offsets to read quantized weights directly.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get tensor data slice by offset and size
    ///
    /// Returns a slice pointing directly into the memory-mapped file.
    /// No data is copied.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from start of file
    /// * `size` - Size in bytes
    ///
    /// # Returns
    ///
    /// Slice of tensor data, or None if out of bounds
    #[must_use]
    pub fn tensor_slice(&self, offset: usize, size: usize) -> Option<&[u8]> {
        let end = offset.checked_add(size)?;
        if end <= self.mmap.len() {
            Some(&self.mmap[offset..end])
        } else {
            None
        }
    }

    /// Get the size of the memory-mapped file
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Advise kernel to prefetch model data sequentially
    ///
    /// Per llama.cpp: Use madvise(MADV_SEQUENTIAL) to hint that the model
    /// will be read sequentially during loading. This improves prefetching.
    #[cfg(unix)]
    pub fn advise_sequential(&self) {
        // MADV_SEQUENTIAL = 2 on Linux
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    /// Advise kernel for random access pattern during inference
    ///
    /// Per llama.cpp: Use madvise(MADV_RANDOM) during inference when
    /// accessing weights non-sequentially.
    #[cfg(unix)]
    pub fn advise_random(&self) {
        // MADV_RANDOM = 1 on Linux
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_RANDOM,
            );
        }
    }

    /// Advise kernel to keep model in memory (reduce swap pressure)
    ///
    /// Per llama.cpp: Use madvise(MADV_WILLNEED) to hint that the model
    /// will be needed soon, triggering prefetch.
    #[cfg(unix)]
    pub fn advise_willneed(&self) {
        // MADV_WILLNEED = 3 on Linux
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Lock model in memory to prevent swapping (requires privileges)
    ///
    /// Per llama.cpp: Use mlock() to ensure model stays in RAM.
    /// Returns true if successful, false if failed (often due to ulimit).
    #[cfg(unix)]
    pub fn lock_memory(&self) -> bool {
        unsafe { libc::mlock(self.mmap.as_ptr().cast::<libc::c_void>(), self.mmap.len()) == 0 }
    }
}

impl GGUFModel {
    /// Parse GGUF file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw GGUF file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid magic number
    /// - Unsupported version
    /// - Malformed data
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header
        let header = Self::parse_header(&mut cursor)?;

        // Parse metadata
        let metadata = Self::parse_metadata(&mut cursor, header.metadata_count)?;

        // Parse tensor info
        let tensors = Self::parse_tensor_info(&mut cursor, header.tensor_count)?;

        // Calculate tensor data start with 32-byte alignment
        let current_pos = cursor.position() as usize;
        let tensor_data_start = current_pos.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;

        Ok(Self {
            header,
            metadata,
            tensors,
            tensor_data_start,
        })
    }

    /// Parse GGUF header
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<GGUFHeader> {
        let mut buf = [0u8; 4];

        // Read magic
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_magic".to_string(),
                reason: e.to_string(),
            })?;
        let magic = u32::from_le_bytes(buf);

        if magic != GGUF_MAGIC {
            return Err(RealizarError::InvalidShape {
                reason: format!("Invalid GGUF magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            });
        }

        // Read version
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_version".to_string(),
                reason: e.to_string(),
            })?;
        let version = u32::from_le_bytes(buf);

        if version != GGUF_VERSION_V3 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_gguf".to_string(),
                reason: format!("Unsupported GGUF version: {version}, only v3 supported"),
            });
        }

        // Read tensor_count
        let mut buf8 = [0u8; 8];
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_tensor_count".to_string(),
                reason: e.to_string(),
            })?;
        let tensor_count = u64::from_le_bytes(buf8);

        // Read metadata_count
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_count".to_string(),
                reason: e.to_string(),
            })?;
        let metadata_count = u64::from_le_bytes(buf8);

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_count,
        })
    }

    /// Parse metadata key-value pairs
    fn parse_metadata(
        cursor: &mut Cursor<&[u8]>,
        count: u64,
    ) -> Result<HashMap<String, GGUFValue>> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            // Read key (string: u64 length + bytes)
            let key = Self::read_string(cursor)?;

            // Read value type (u32)
            let mut buf = [0u8; 4];
            cursor
                .read_exact(&mut buf)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "read_metadata_type".to_string(),
                    reason: e.to_string(),
                })?;
            let value_type = u32::from_le_bytes(buf);

            // Read value based on type
            let value = Self::read_value(cursor, value_type)?;

            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    /// Read a string: u64 length + UTF-8 bytes
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let mut buf8 = [0u8; 8];
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_string_length".to_string(),
                reason: e.to_string(),
            })?;
        let len_u64 = u64::from_le_bytes(buf8);
        let len = usize::try_from(len_u64).map_err(|_| RealizarError::UnsupportedOperation {
            operation: "convert_string_length".to_string(),
            reason: format!("String length {len_u64} exceeds platform usize limit"),
        })?;

        let mut string_bytes = vec![0u8; len];
        cursor
            .read_exact(&mut string_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_string_data".to_string(),
                reason: e.to_string(),
            })?;

        String::from_utf8(string_bytes).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "parse_utf8".to_string(),
            reason: e.to_string(),
        })
    }

    /// Read a value based on type
    fn read_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<GGUFValue> {
        match value_type {
            0 => Ok(GGUFValue::UInt8(Self::read_u8(cursor)?)),
            1 => Ok(GGUFValue::Int8(Self::read_i8(cursor)?)),
            2 => Ok(GGUFValue::UInt16(Self::read_u16(cursor)?)),
            3 => Ok(GGUFValue::Int16(Self::read_i16(cursor)?)),
            4 => Ok(GGUFValue::UInt32(Self::read_u32(cursor)?)),
            5 => Ok(GGUFValue::Int32(Self::read_i32(cursor)?)),
            6 => Ok(GGUFValue::Float32(Self::read_f32(cursor)?)),
            7 => Ok(GGUFValue::Bool(Self::read_bool(cursor)?)),
            8 => Ok(GGUFValue::String(Self::read_string(cursor)?)),
            9 => {
                // Array: element_type (u32) + array_len (u64) + elements
                let element_type = Self::read_u32(cursor)?;
                let array_len = Self::read_u64(cursor)?;

                // Safely convert array_len to usize
                let len = usize::try_from(array_len).map_err(|_| RealizarError::InvalidShape {
                    reason: format!("Array length too large: {array_len}"),
                })?;

                let mut elements = Vec::with_capacity(len);
                for _ in 0..array_len {
                    elements.push(Self::read_value(cursor, element_type)?);
                }
                Ok(GGUFValue::Array(elements))
            },
            10 => Ok(GGUFValue::UInt64(Self::read_u64(cursor)?)),
            11 => Ok(GGUFValue::Int64(Self::read_i64(cursor)?)),
            12 => Ok(GGUFValue::Float64(Self::read_f64(cursor)?)),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "read_value".to_string(),
                reason: format!("Unsupported value type: {value_type}"),
            }),
        }
    }

    /// Read u8
    fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
        let mut buf = [0u8; 1];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_u8".to_string(),
                reason: e.to_string(),
            })?;
        Ok(buf[0])
    }

    /// Read i8
    fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
        let mut buf = [0u8; 1];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_i8".to_string(),
                reason: e.to_string(),
            })?;
        Ok(i8::from_le_bytes(buf))
    }

    /// Read u16
    fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        let mut buf = [0u8; 2];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_u16".to_string(),
                reason: e.to_string(),
            })?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read i16
    fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
        let mut buf = [0u8; 2];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_i16".to_string(),
                reason: e.to_string(),
            })?;
        Ok(i16::from_le_bytes(buf))
    }

    /// Read u32
    fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
        let mut buf = [0u8; 4];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_u32".to_string(),
                reason: e.to_string(),
            })?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read i32
    fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
        let mut buf = [0u8; 4];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_i32".to_string(),
                reason: e.to_string(),
            })?;
        Ok(i32::from_le_bytes(buf))
    }

    /// Read f32
    fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
        let mut buf = [0u8; 4];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_f32".to_string(),
                reason: e.to_string(),
            })?;
        Ok(f32::from_le_bytes(buf))
    }

    /// Read bool
    fn read_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool> {
        let mut buf = [0u8; 1];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_bool".to_string(),
                reason: e.to_string(),
            })?;
        Ok(buf[0] != 0)
    }

    /// Read u64
    fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_u64".to_string(),
                reason: e.to_string(),
            })?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read i64
    fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_i64".to_string(),
                reason: e.to_string(),
            })?;
        Ok(i64::from_le_bytes(buf))
    }

    /// Read f64
    fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_f64".to_string(),
                reason: e.to_string(),
            })?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Parse tensor info
    fn parse_tensor_info(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::new();

        for _ in 0..count {
            // Read tensor name (string)
            let name = Self::read_string(cursor)?;

            // Read n_dims (u32)
            let n_dims = Self::read_u32(cursor)?;

            // Read dimensions array
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(cursor)?);
            }

            // Read quantization type (u32)
            let qtype = Self::read_u32(cursor)?;

            // Read offset (u64)
            let offset = Self::read_u64(cursor)?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                qtype,
                offset,
            });
        }

        Ok(tensors)
    }

    /// Extract tensor data by name with dequantization
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    /// * `file_data` - Complete GGUF file bytes
    ///
    /// # Returns
    ///
    /// Dequantized f32 tensor data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Unsupported quantization type
    /// - Invalid data at offset
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let file_data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&file_data)?;
    /// let weights = model.get_tensor_f32("layer.0.weight", &file_data)?;
    /// ```
    pub fn get_tensor_f32(&self, name: &str, file_data: &[u8]) -> Result<Vec<f32>> {
        // Find tensor info
        let tensor = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Calculate tensor size in elements
        let size: usize = tensor
            .dims
            .iter()
            .try_fold(1usize, |acc, &dim| {
                usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d))
            })
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor dimensions overflow: {:?}", tensor.dims),
            })?;

        // Convert tensor offset to usize and add tensor data start
        let tensor_offset =
            usize::try_from(tensor.offset).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_offset".to_string(),
                reason: format!("Offset {} exceeds platform usize limit", tensor.offset),
            })?;
        let offset = self.tensor_data_start + tensor_offset;

        // Extract and dequantize based on qtype
        match tensor.qtype {
            GGUF_TYPE_F32 => {
                // Unquantized F32 data
                let byte_size = size * 4; // 4 bytes per f32
                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(values)
            },
            GGUF_TYPE_Q4_0 => {
                // Q4_0 quantized data
                use crate::quantize::dequantize_q4_0;

                // Q4_0 block size: 20 bytes (4 for scale + 16 for quants)
                const BLOCK_BYTES: usize = 20;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_0(bytes)?;

                // Trim to exact size (dequantization pads to block boundaries)
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q8_0 => {
                // Q8_0 quantized data - use SIMD-parallel for faster loading
                use crate::quantize::dequantize_q8_0_simd;

                // Q8_0 block size: 36 bytes (4 for scale + 32 for quants)
                const BLOCK_BYTES: usize = 36;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q8_0_simd(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q4_K => {
                // Q4_K quantized data (K-quantization) - use SIMD-parallel for faster loading
                use crate::quantize::{dequantize_q4_k_simd, QK_K};

                // Q4_K super-block size: 144 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 144;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_k_simd(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_K => {
                // Q5_K quantized data (K-quantization)
                use crate::quantize::{dequantize_q5_k, QK_K};

                // Q5_K super-block size: 176 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 176;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_k(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q6_K => {
                // Q6_K quantized data (K-quantization)
                use crate::quantize::{dequantize_q6_k, QK_K};

                // Q6_K super-block size: 210 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 210;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q6_k(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_F16 => {
                // F16 (half-precision float) data
                use crate::quantize::dequantize_f16;

                let byte_size = size * 2; // 2 bytes per f16
                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let values = dequantize_f16(bytes)?;
                Ok(values)
            },
            GGUF_TYPE_Q4_1 => {
                // Q4_1 quantized data
                use crate::quantize::dequantize_q4_1;

                // Q4_1 block size: 20 bytes (2 for scale + 2 for min + 16 for quants)
                const BLOCK_BYTES: usize = 20;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_1(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_0 => {
                // Q5_0 quantized data
                use crate::quantize::dequantize_q5_0;

                // Q5_0 block size: 22 bytes (2 for scale + 4 for high bits + 16 for quants)
                const BLOCK_BYTES: usize = 22;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_0(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_1 => {
                // Q5_1 quantized data
                use crate::quantize::dequantize_q5_1;

                // Q5_1 block size: 24 bytes (2 for scale + 2 for min + 4 for high bits + 16 for quants)
                const BLOCK_BYTES: usize = 24;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

                if offset + byte_size > file_data.len() {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "get_tensor_f32".to_string(),
                        reason: format!(
                            "Data range [{}, {}) exceeds file size {}",
                            offset,
                            offset + byte_size,
                            file_data.len()
                        ),
                    });
                }

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_1(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Unsupported quantization type: {}", tensor.qtype),
            }),
        }
    }

    /// Extract model architecture from metadata
    pub fn architecture(&self) -> Option<&str> {
        if let Some(GGUFValue::String(arch)) = self.metadata.get("general.architecture") {
            Some(arch.as_str())
        } else {
            None
        }
    }

    /// Get embedding dimension from metadata
    pub fn embedding_dim(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = format!("{}.embedding_length", arch);
        if let Some(GGUFValue::UInt32(dim)) = self.metadata.get(&key) {
            Some(*dim as usize)
        } else {
            None
        }
    }

    /// Get number of layers from metadata
    pub fn num_layers(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = format!("{}.block_count", arch);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get number of attention heads from metadata
    pub fn num_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.head_count", arch);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get context length from metadata
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = format!("{}.context_length", arch);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }
}

/// Configuration for GGUF transformer inference
#[derive(Debug, Clone)]
pub struct GGUFConfig {
    /// Model architecture (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding dimension (hidden size)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA, often num_heads or num_heads/8)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Context length
    pub context_length: usize,
    /// RoPE theta (position encoding base)
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl GGUFConfig {
    /// Extract configuration from GGUF model metadata
    ///
    /// # Errors
    ///
    /// Returns an error if required metadata fields are missing from the GGUF model.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let architecture = model
            .architecture()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing general.architecture in GGUF metadata".to_string(),
            })?
            .to_string();

        let hidden_dim = model
            .embedding_dim()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing embedding_length in GGUF metadata".to_string(),
            })?;

        let num_layers = model
            .num_layers()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing block_count in GGUF metadata".to_string(),
            })?;

        // Try to get num_heads, default based on hidden_dim if not found
        let num_heads = model.num_heads().unwrap_or(hidden_dim / 64);

        // Get vocab_size from token_embd tensor
        let vocab_size = model
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map_or(32000, |t| t.dims.get(1).copied().unwrap_or(32000) as usize);

        // Infer intermediate_dim from ffn_up tensor
        let intermediate_dim = model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight")
            .map_or(hidden_dim * 4, |t| {
                t.dims.get(1).copied().unwrap_or(hidden_dim as u64 * 4) as usize
            });

        let context_length = model.context_length().unwrap_or(2048);

        // Default rope_theta for most models
        let rope_theta = 10000.0;
        let eps = 1e-5;

        // num_kv_heads (for GQA, usually same as num_heads or num_heads/8)
        let num_kv_heads = num_heads;

        Ok(Self {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
        })
    }
}

/// GGUF Transformer for inference
///
/// Holds loaded weights and configuration for transformer inference.
/// Supports phi-2, llama, qwen2, and similar architectures.
pub struct GGUFTransformer {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embedding weights [vocab_size, hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Attention weights per layer
    pub layers: Vec<GGUFTransformerLayer>,
    /// Output norm weight
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head / output projection weight
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional)
    pub lm_head_bias: Option<Vec<f32>>,
}

/// Weights for a single transformer layer
pub struct GGUFTransformerLayer {
    /// Attention norm weight
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (combined for phi-2, concatenated Q+K+V for llama)
    pub qkv_weight: Vec<f32>,
    /// QKV bias (phi-2 has bias, llama doesn't)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate projection weight (SwiGLU models like llama)
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate projection bias
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (for models with separate FFN normalization)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias
    pub ffn_norm_bias: Option<Vec<f32>>,
}

#[allow(clippy::unused_self)]
#[allow(clippy::similar_names)]
impl GGUFTransformer {
    /// Load transformer weights from GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model
    /// * `file_data` - Original file bytes for tensor extraction
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or malformed
    pub fn from_gguf(model: &GGUFModel, file_data: &[u8]) -> Result<Self> {
        let config = GGUFConfig::from_gguf(model)?;

        // Load token embedding
        let token_embedding = model.get_tensor_f32("token_embd.weight", file_data)?;

        // Load layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_layer(model, file_data, layer_idx)?;
            layers.push(layer);
        }

        // Load output norm
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", file_data)?;
        let output_norm_bias = model.get_tensor_f32("output_norm.bias", file_data).ok();

        // Load LM head (output projection)
        let lm_head_weight = model.get_tensor_f32("output.weight", file_data)?;
        let lm_head_bias = model.get_tensor_f32("output.bias", file_data).ok();

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// Load a single transformer layer
    ///
    /// Supports both tensor naming conventions:
    /// - phi-2 style: combined `attn_qkv.weight`
    /// - llama style: separate `attn_q.weight`, `attn_k.weight`, `attn_v.weight`
    fn load_layer(
        model: &GGUFModel,
        file_data: &[u8],
        layer_idx: usize,
    ) -> Result<GGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), file_data)?;
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), file_data)
            .ok();

        // QKV weights - try combined first (phi-2), fall back to separate (llama)
        let (qkv_weight, qkv_bias) = if let Ok(combined) =
            model.get_tensor_f32(&format!("{}.attn_qkv.weight", prefix), file_data)
        {
            // phi-2 style: combined QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), file_data)
                .ok();
            (combined, bias)
        } else {
            // llama style: separate Q, K, V tensors - concatenate them
            let q_weight = model.get_tensor_f32(&format!("{}.attn_q.weight", prefix), file_data)?;
            let k_weight = model.get_tensor_f32(&format!("{}.attn_k.weight", prefix), file_data)?;
            let v_weight = model.get_tensor_f32(&format!("{}.attn_v.weight", prefix), file_data)?;

            // Concatenate Q, K, V weights
            let mut qkv = Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
            qkv.extend_from_slice(&q_weight);
            qkv.extend_from_slice(&k_weight);
            qkv.extend_from_slice(&v_weight);

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), file_data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), file_data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), file_data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(q), Some(k), Some(v)) => {
                    let mut combined_bias = Vec::with_capacity(q.len() + k.len() + v.len());
                    combined_bias.extend_from_slice(&q);
                    combined_bias.extend_from_slice(&k);
                    combined_bias.extend_from_slice(&v);
                    Some(combined_bias)
                },
                _ => None,
            };

            (qkv, bias)
        };

        // Attention output
        let attn_output_weight =
            model.get_tensor_f32(&format!("{}.attn_output.weight", prefix), file_data)?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), file_data)
            .ok();

        // FFN gate (SwiGLU models like llama have this)
        let ffn_gate_weight = model
            .get_tensor_f32(&format!("{}.ffn_gate.weight", prefix), file_data)
            .ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), file_data)
            .ok();

        // FFN up/down projections
        let ffn_up_weight =
            model.get_tensor_f32(&format!("{}.ffn_up.weight", prefix), file_data)?;
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), file_data)
            .ok();
        let ffn_down_weight =
            model.get_tensor_f32(&format!("{}.ffn_down.weight", prefix), file_data)?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), file_data)
            .ok();

        // FFN norm (models with separate FFN normalization)
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), file_data)
            .ok();
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), file_data)
            .ok();

        Ok(GGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
            ffn_norm_weight,
            ffn_norm_bias,
        })
    }

    /// Look up token embeddings
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to look up
    ///
    /// # Returns
    ///
    /// Embedding matrix [seq_len, hidden_dim]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                // Pad with zeros for out-of-bounds tokens
                embeddings.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        embeddings
    }

    /// Apply layer normalization
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let hidden_dim = weight.len();
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for i in 0..seq_len {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            let x = &input[start..end];

            // Compute mean
            let mean: f32 = x.iter().sum::<f32>() / hidden_dim as f32;

            // Compute variance
            let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_dim as f32;

            // Normalize and apply affine transform
            let inv_std = (var + eps).sqrt().recip();
            for j in 0..hidden_dim {
                let normalized = (x[j] - mean) * inv_std;
                let mut val = normalized * weight[j];
                if let Some(b) = bias {
                    val += b[j];
                }
                output.push(val);
            }
        }

        output
    }

    /// Matrix-vector multiplication (for inference with batch_size=1)
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let mut output = Vec::with_capacity(seq_len * out_dim);

        for s in 0..seq_len {
            let x_start = s * in_dim;
            for o in 0..out_dim {
                let mut sum = 0.0f32;
                for i in 0..in_dim {
                    // Weight layout: [in_dim, out_dim] row-major
                    // w[i][o] = weight[i * out_dim + o]
                    sum += input[x_start + i] * weight[i * out_dim + o];
                }
                output.push(sum);
            }
        }

        output
    }

    /// Add bias to output
    fn add_bias(&self, output: &mut [f32], bias: &[f32]) {
        let out_dim = bias.len();
        let seq_len = output.len() / out_dim;
        for s in 0..seq_len {
            for o in 0..out_dim {
                output[s * out_dim + o] += bias[o];
            }
        }
    }

    /// Apply GELU activation
    fn gelu(&self, input: &mut [f32]) {
        for x in input.iter_mut() {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let sqrt_2_over_pi = 0.797_884_6_f32;
            let c = 0.044_715_f32;
            let inner = sqrt_2_over_pi * (*x + c * *x * *x * *x);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Simple forward pass for next-token prediction
    ///
    /// This is a simplified forward pass without KV caching or RoPE,
    /// suitable for testing and simple use cases.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail during the forward pass.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            // For phi-2: qkv_weight is [hidden_dim, 3*hidden_dim]
            let qkv_dim = 3 * hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. For simplicity, skip actual attention and just use averaged QKV
            // (Real attention would need RoPE, causal masking, and proper attention)
            // Here we do a very simplified version for testing
            let seq_len = token_ids.len();
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);

            // Average Q, K, V and project through attention output
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                // Extract Q for this position (simplified)
                for h in 0..hidden_dim {
                    attn_out.push(qkv[qkv_start + h]); // Just use Q for now
                }
            }

            // 2d. Attention output projection
            let mut attn_output =
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN (for phi-2, no separate ffn_norm, uses same norm)
            // FFN up projection
            let mut ffn_hidden =
                self.matmul(&hidden, &layer.ffn_up_weight, hidden_dim, intermediate_dim);
            if let Some(ref bias) = layer.ffn_up_bias {
                self.add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation
            self.gelu(&mut ffn_hidden);

            // FFN down projection
            let mut ffn_output = self.matmul(
                &ffn_hidden,
                &layer.ffn_down_weight,
                intermediate_dim,
                hidden_dim,
            );
            if let Some(ref bias) = layer.ffn_down_bias {
                self.add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            if layer_idx == 0 {
                // Print first layer stats for debugging
                let min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
                let max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean: f32 = hidden.iter().sum::<f32>() / hidden.len() as f32;
                eprintln!(
                    "Layer 0 output: min={:.4}, max={:.4}, mean={:.4}",
                    min, max, mean
                );
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only for last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = Vec::with_capacity(self.config.vocab_size);
        for o in 0..self.config.vocab_size {
            let sum: f32 = last_hidden
                .iter()
                .enumerate()
                .map(|(i, &h)| h * self.lm_head_weight[i * self.config.vocab_size + o])
                .sum();
            let final_val = if let Some(ref bias) = self.lm_head_bias {
                sum + bias[o]
            } else {
                sum
            };
            logits.push(final_val);
        }

        Ok(logits)
    }

    /// Get the most likely next token
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(max_idx as u32)
    }
}

// ============================================================================
// Quantized Transformer (Fused Operations)
// ============================================================================

/// Reference to a quantized tensor stored in memory-mapped file
///
/// Per Wulf & McKee (1995) "Hitting the Memory Wall", memory bandwidth is the
/// bottleneck for LLM inference. By keeping weights in quantized form and
/// dequantizing inline during computation, we achieve 8x memory bandwidth
/// reduction for Q4_K format.
#[derive(Debug, Clone)]
pub struct QuantizedTensorRef {
    /// Byte offset in file where tensor data starts
    pub offset: usize,
    /// Size in bytes of the quantized data
    pub byte_size: usize,
    /// Number of elements after dequantization
    pub num_elements: usize,
    /// Quantization type (GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K, etc.)
    pub qtype: u32,
}

/// Quantized transformer layer weights (stored as byte references)
///
/// Unlike `GGUFTransformerLayer` which stores dequantized Vec<f32>,
/// this stores references to quantized data for fused operations.
pub struct QuantizedGGUFTransformerLayer {
    /// Attention norm weight (kept as f32 - small, read once per token)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (quantized)
    pub qkv_weight: QuantizedTensorRef,
    /// QKV bias (optional, f32)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_output_weight: QuantizedTensorRef,
    /// Attention output bias (optional, f32)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN up projection (quantized)
    pub ffn_up_weight: QuantizedTensorRef,
    /// FFN up bias (optional, f32)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection (quantized)
    pub ffn_down_weight: QuantizedTensorRef,
    /// FFN down bias (optional, f32)
    pub ffn_down_bias: Option<Vec<f32>>,
}

/// Quantized GGUF Transformer for fused inference
///
/// Per Williams et al. (2009) roofline model, LLM inference is memory-bound.
/// This transformer stores weights in quantized form and uses fused
/// dequant+dot operations to minimize memory bandwidth.
///
/// # Performance Benefits
///
/// - **8x bandwidth reduction** for Q4_K vs f32 (144 bytes vs 1024 bytes per 256 values)
/// - **Zero intermediate buffers** - dequantization happens inline with dot product
/// - **SIMD acceleration** - AVX2/FMA fused operations when available
/// - **Zero-copy loading** - weights stay in memory-mapped file
///
/// # Architecture
///
/// ```text
/// [Memory-mapped Q4_K bytes] → [fused_q4k_dot_simd] → [f32 result]
///                               ↑
///                         No intermediate Vec<f32>!
/// ```
pub struct QuantizedGGUFTransformer<'a> {
    /// Model configuration
    pub config: GGUFConfig,
    /// Reference to memory-mapped file data
    pub data: &'a [u8],
    /// Token embedding (kept as f32 for lookup)
    pub token_embedding: Vec<f32>,
    /// Quantized layer weights
    pub layers: Vec<QuantizedGGUFTransformerLayer>,
    /// Output norm weight (f32)
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (quantized for large vocab)
    pub lm_head_weight: QuantizedTensorRef,
    /// LM head bias (optional, f32)
    pub lm_head_bias: Option<Vec<f32>>,
}

impl<'a> QuantizedGGUFTransformer<'a> {
    /// Load quantized transformer from memory-mapped GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model metadata
    /// * `data` - Memory-mapped file data (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or have unsupported format
    pub fn from_gguf(model: &GGUFModel, data: &'a [u8]) -> Result<Self> {
        let config = GGUFConfig::from_gguf(model)?;

        // Token embedding - keep as f32 for efficient lookup
        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;

        // Load layers with quantized weight references
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_quantized_layer(model, data, layer_idx)?;
            layers.push(layer);
        }

        // Output norm - small, keep as f32
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;
        let output_norm_bias = model.get_tensor_f32("output_norm.bias", data).ok();

        // LM head - large, keep quantized
        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")?;
        let lm_head_bias = model.get_tensor_f32("output.bias", data).ok();

        Ok(Self {
            config,
            data,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// Get tensor reference (offset + size + qtype) without dequantization
    fn get_tensor_ref(model: &GGUFModel, data: &[u8], name: &str) -> Result<QuantizedTensorRef> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor '{}' not found", name),
            })?;

        let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
        let offset = model.tensor_data_start + tensor.offset as usize;

        // Calculate byte size based on quantization type
        let byte_size = match tensor.qtype {
            GGUF_TYPE_F32 => num_elements * 4,
            GGUF_TYPE_Q4_0 => {
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 20;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q8_0 => {
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 36;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q4_K => {
                use crate::quantize::QK_K;
                const SUPER_BLOCK_BYTES: usize = 144;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q5_K => {
                use crate::quantize::QK_K;
                const SUPER_BLOCK_BYTES: usize = 176;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q6_K => {
                use crate::quantize::QK_K;
                const SUPER_BLOCK_BYTES: usize = 210;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            _ => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "get_tensor_ref".to_string(),
                    reason: format!("Unsupported quantization type: {}", tensor.qtype),
                });
            },
        };

        // Validate bounds
        if offset + byte_size > data.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Tensor '{}' data range [{}, {}) exceeds file size {}",
                    name,
                    offset,
                    offset + byte_size,
                    data.len()
                ),
            });
        }

        Ok(QuantizedTensorRef {
            offset,
            byte_size,
            num_elements,
            qtype: tensor.qtype,
        })
    }

    /// Load a single quantized transformer layer
    fn load_quantized_layer(
        model: &GGUFModel,
        data: &[u8],
        layer_idx: usize,
    ) -> Result<QuantizedGGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm - small, keep as f32
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
            .ok();

        // QKV - large, keep quantized
        let qkv_weight = Self::get_tensor_ref(model, data, &format!("{}.attn_qkv.weight", prefix))?;
        let qkv_bias = model
            .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
            .ok();

        // Attention output - large, keep quantized
        let attn_output_weight =
            Self::get_tensor_ref(model, data, &format!("{}.attn_output.weight", prefix))?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
            .ok();

        // FFN - large, keep quantized
        let ffn_up_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_up.weight", prefix))?;
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
            .ok();
        let ffn_down_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_down.weight", prefix))?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
            .ok();

        Ok(QuantizedGGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        })
    }

    /// Get tensor data slice from memory-mapped file
    #[inline]
    fn tensor_data(&self, tensor_ref: &QuantizedTensorRef) -> &[u8] {
        &self.data[tensor_ref.offset..tensor_ref.offset + tensor_ref.byte_size]
    }

    /// Fused quantized matrix-vector multiply with parallel processing (Phase 2+3)
    ///
    /// Performs dequantization inline with dot product - NO intermediate buffer.
    /// Uses rayon parallel iterators per Blumofe & Leiserson [6] for multi-core acceleration.
    ///
    /// Supports Q4_K, Q5_K, and Q6_K with fused operations.
    fn fused_matmul(
        &self,
        input: &[f32],
        weight_ref: &QuantizedTensorRef,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::{
            fused_q4k_parallel_matvec, fused_q5k_parallel_matvec, fused_q6k_parallel_matvec,
        };

        let seq_len = input.len() / in_dim;
        let weight_data = self.tensor_data(weight_ref);

        // For sequence length > 1, process each position
        if seq_len > 1 {
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = match weight_ref.qtype {
                    GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(weight_data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(weight_data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(weight_data, x, in_dim, out_dim)?,
                    _ => {
                        return Err(RealizarError::UnsupportedOperation {
                            operation: "fused_matmul".to_string(),
                            reason: format!(
                                "Fused matmul only supports Q4_K/Q5_K/Q6_K, got type {}",
                                weight_ref.qtype
                            ),
                        });
                    },
                };
                output.extend_from_slice(&row_output);
            }
            Ok(output)
        } else {
            // Single position - use parallel matvec directly
            match weight_ref.qtype {
                GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(weight_data, input, in_dim, out_dim),
                GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(weight_data, input, in_dim, out_dim),
                GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(weight_data, input, in_dim, out_dim),
                _ => Err(RealizarError::UnsupportedOperation {
                    operation: "fused_matmul".to_string(),
                    reason: format!(
                        "Fused matmul only supports Q4_K/Q5_K/Q6_K, got type {}",
                        weight_ref.qtype
                    ),
                }),
            }
        }
    }

    /// Look up token embeddings
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                embeddings.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        embeddings
    }

    /// Apply layer normalization
    #[allow(clippy::unused_self)]
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let hidden_dim = weight.len();
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for i in 0..seq_len {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            let x = &input[start..end];

            let mean: f32 = x.iter().sum::<f32>() / hidden_dim as f32;
            let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
            let inv_std = (var + eps).sqrt().recip();

            for j in 0..hidden_dim {
                let normalized = (x[j] - mean) * inv_std;
                let mut val = normalized * weight[j];
                if let Some(b) = bias {
                    val += b[j];
                }
                output.push(val);
            }
        }

        output
    }

    /// Add bias to output
    #[allow(clippy::unused_self)]
    fn add_bias(&self, output: &mut [f32], bias: &[f32]) {
        let out_dim = bias.len();
        let seq_len = output.len() / out_dim;
        for s in 0..seq_len {
            for o in 0..out_dim {
                output[s * out_dim + o] += bias[o];
            }
        }
    }

    /// Apply GELU activation
    #[allow(clippy::unused_self)]
    fn gelu(&self, input: &mut [f32]) {
        for x in input.iter_mut() {
            let sqrt_2_over_pi = 0.797_884_6_f32;
            let c = 0.044_715_f32;
            let inner = sqrt_2_over_pi * (*x + c * *x * *x * *x);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Forward pass with fused quantized operations
    ///
    /// This is the optimized forward pass that keeps weights in quantized form
    /// and uses fused dequant+dot operations to minimize memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup (f32, fast)
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers with fused ops
        for layer in &self.layers {
            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection with FUSED dequant+dot
            let qkv_dim = 3 * hidden_dim;
            let mut qkv = self.fused_matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim)?;
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Simplified attention (real impl would have RoPE, causal mask, etc.)
            let seq_len = token_ids.len();
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                for h in 0..hidden_dim {
                    attn_out.push(qkv[qkv_start + h]);
                }
            }

            // 2d. Attention output projection with FUSED dequant+dot
            let mut attn_output =
                self.fused_matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)?;
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection with FUSED dequant+dot
            let mut ffn_hidden =
                self.fused_matmul(&hidden, &layer.ffn_up_weight, hidden_dim, intermediate_dim)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                self.add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation
            self.gelu(&mut ffn_hidden);

            // 2g. FFN down projection with FUSED dequant+dot
            let mut ffn_output = self.fused_matmul(
                &ffn_hidden,
                &layer.ffn_down_weight,
                intermediate_dim,
                hidden_dim,
            )?;
            if let Some(ref bias) = layer.ffn_down_bias {
                self.add_bias(&mut ffn_output, bias);
            }

            // Residual connection
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

        // 4. LM head projection with FUSED dequant+dot (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Compute logits using fused op
        let mut logits = self.fused_matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        )?;

        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Get the most likely next token
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(max_idx as u32)
    }

    /// Generate a sequence of tokens
    ///
    /// This is the end-to-end generation loop that uses fused Q4_K operations.
    /// Per benchmark-model-runners-spec.md "What's Remaining" item 1.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn generate(&self, prompt: &[u32], config: &QuantizedGenerateConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();
        let max_len = prompt.len() + config.max_tokens;

        for _ in 0..config.max_tokens {
            // Forward pass with fused Q4_K ops
            let logits = self.forward(&tokens)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                Self::argmax(&logits)
            } else {
                // Temperature + top-k sampling
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_len {
                break;
            }
        }

        Ok(tokens)
    }

    /// Greedy argmax over logits
    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }

    /// Top-k sampling with temperature
    fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        // Softmax over top-k
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
        let probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
            .collect();

        // Sample from distribution (deterministic for now via cumulative)
        // Use a simple hash-based pseudo-random for reproducibility
        let hash = logits.len() as u32 ^ (top_k as u32) ^ ((temperature * 1000.0) as u32);
        let r = (hash % 1000) as f32 / 1000.0;
        let mut cumsum = 0.0;
        for (idx, prob) in &probs {
            cumsum += prob;
            if cumsum >= r {
                return *idx as u32;
            }
        }
        probs.last().map_or(0, |(idx, _)| *idx as u32)
    }
}

/// Configuration for quantized generation
///
/// Per benchmark-model-runners-spec.md "What's Remaining" item 1:
/// End-to-end Q4_K inference with generation config.
#[derive(Debug, Clone)]
pub struct QuantizedGenerateConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    pub top_k: usize,
    /// Stop token IDs
    pub stop_tokens: Vec<u32>,
}

impl Default for QuantizedGenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }
}

impl QuantizedGenerateConfig {
    /// Create config for deterministic (greedy) generation
    #[must_use]
    pub fn deterministic(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set stop tokens
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<u32>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_constant() {
        // "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        // Verify it spells "GGUF"
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn test_parse_valid_header() {
        // Minimal valid GGUF v3 header
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, 3);
        assert_eq!(model.header.tensor_count, 0);
        assert_eq!(model.header.metadata_count, 0);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"BAAD"); // Invalid magic
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::InvalidShape { .. }
        ));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::UnsupportedOperation { .. }
        ));
    }

    #[test]
    fn test_truncated_data() {
        // Only 4 bytes (magic only)
        let data = b"GGUF";
        let result = GGUFModel::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let data = &[];
        let result = GGUFModel::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_uint32_metadata() {
        // GGUF header with 1 metadata item (UInt32)
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata: key = "test.value", value_type = UInt32 (4), value = 42
        let key = "test.value";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length
        data.extend_from_slice(key.as_bytes()); // key string
        data.extend_from_slice(&4u32.to_le_bytes()); // value_type = UInt32
        data.extend_from_slice(&42u32.to_le_bytes()); // value = 42

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 1);
        assert_eq!(
            model.metadata.get("test.value"),
            Some(&GGUFValue::UInt32(42))
        );
    }

    #[test]
    fn test_parse_string_metadata() {
        // GGUF header with 1 metadata item (String)
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata: key = "model.name", value_type = String (8), value = "TestModel"
        let key = "model.name";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&8u32.to_le_bytes()); // value_type = String
        let value = "TestModel";
        data.extend_from_slice(&(value.len() as u64).to_le_bytes()); // string length
        data.extend_from_slice(value.as_bytes()); // string data

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 1);
        assert_eq!(
            model.metadata.get("model.name"),
            Some(&GGUFValue::String("TestModel".to_string()))
        );
    }

    #[test]
    fn test_parse_multiple_metadata() {
        // GGUF header with 2 metadata items
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // metadata_count = 2

        // First: key = "version", value = UInt32(1)
        data.extend_from_slice(&7u64.to_le_bytes());
        data.extend_from_slice(b"version");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        // Second: key = "arch", value = String("llama")
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(b"arch");
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"llama");

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 2);
        assert_eq!(model.metadata.get("version"), Some(&GGUFValue::UInt32(1)));
        assert_eq!(
            model.metadata.get("arch"),
            Some(&GGUFValue::String("llama".to_string()))
        );
    }

    #[test]
    fn test_parse_single_tensor_info() {
        // GGUF header with 1 tensor
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor: name = "weight", n_dims = 2, dims = [128, 256], qtype = 0, offset = 1024
        let name = "weight";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        data.extend_from_slice(&128u64.to_le_bytes()); // dim[0] = 128
        data.extend_from_slice(&256u64.to_le_bytes()); // dim[1] = 256
        data.extend_from_slice(&0u32.to_le_bytes()); // qtype = 0
        data.extend_from_slice(&1024u64.to_le_bytes()); // offset = 1024

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.tensors.len(), 1);
        let tensor = &model.tensors[0];
        assert_eq!(tensor.name, "weight");
        assert_eq!(tensor.n_dims, 2);
        assert_eq!(tensor.dims, vec![128, 256]);
        assert_eq!(tensor.qtype, 0);
        assert_eq!(tensor.offset, 1024);
    }

    #[test]
    fn test_parse_tensor_3d() {
        // GGUF header with 1 tensor (3D)
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor: name = "conv.weight", n_dims = 3, dims = [64, 64, 3]
        let name = "conv.weight";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // n_dims = 3
        data.extend_from_slice(&64u64.to_le_bytes());
        data.extend_from_slice(&64u64.to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // qtype = 2 (quantized)
        data.extend_from_slice(&2048u64.to_le_bytes()); // offset = 2048

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.tensors.len(), 1);
        let tensor = &model.tensors[0];
        assert_eq!(tensor.name, "conv.weight");
        assert_eq!(tensor.n_dims, 3);
        assert_eq!(tensor.dims, vec![64, 64, 3]);
        assert_eq!(tensor.qtype, 2);
        assert_eq!(tensor.offset, 2048);
    }

    #[test]
    fn test_parse_metadata_and_tensors() {
        // GGUF with both metadata and tensors
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata: model.type = String("llama")
        data.extend_from_slice(&10u64.to_le_bytes());
        data.extend_from_slice(b"model.type");
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"llama");

        // Tensor: embedding
        data.extend_from_slice(&9u64.to_le_bytes());
        data.extend_from_slice(b"embedding");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&32000u64.to_le_bytes());
        data.extend_from_slice(&4096u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 1);
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(
            model.metadata.get("model.type"),
            Some(&GGUFValue::String("llama".to_string()))
        );
        assert_eq!(model.tensors[0].name, "embedding");
    }

    #[test]
    fn test_parse_uint8_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata: key = "byte_val", value_type = UInt8 (0), value = 255
        let key = "byte_val";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // value_type = UInt8
        data.push(255u8); // value = 255

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.get("byte_val"), Some(&GGUFValue::UInt8(255)));
    }

    #[test]
    fn test_parse_int8_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "signed_byte";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // value_type = Int8
        data.extend_from_slice(&(-42i8).to_le_bytes()); // value = -42

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("signed_byte"),
            Some(&GGUFValue::Int8(-42))
        );
    }

    #[test]
    fn test_parse_uint16_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "short_val";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // value_type = UInt16
        data.extend_from_slice(&65535u16.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("short_val"),
            Some(&GGUFValue::UInt16(65535))
        );
    }

    #[test]
    fn test_parse_int16_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "signed_short";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // value_type = Int16
        data.extend_from_slice(&(-1000i16).to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("signed_short"),
            Some(&GGUFValue::Int16(-1000))
        );
    }

    #[test]
    fn test_parse_int32_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "signed_int";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&5u32.to_le_bytes()); // value_type = Int32
        data.extend_from_slice(&(-100_000_i32).to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("signed_int"),
            Some(&GGUFValue::Int32(-100_000))
        );
    }

    #[test]
    fn test_parse_float32_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "float_val";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&6u32.to_le_bytes()); // value_type = Float32
        data.extend_from_slice(&1.25f32.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        if let Some(GGUFValue::Float32(val)) = model.metadata.get("float_val") {
            assert!((val - 1.25).abs() < 1e-5);
        } else {
            panic!("Expected Float32 value");
        }
    }

    #[test]
    fn test_parse_bool_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "is_enabled";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&7u32.to_le_bytes()); // value_type = Bool
        data.push(1u8); // true

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("is_enabled"),
            Some(&GGUFValue::Bool(true))
        );
    }

    #[test]
    fn test_parse_bool_false_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "is_disabled";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&7u32.to_le_bytes()); // value_type = Bool
        data.push(0u8); // false

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("is_disabled"),
            Some(&GGUFValue::Bool(false))
        );
    }

    #[test]
    fn test_parse_uint64_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "big_uint";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&10u32.to_le_bytes()); // value_type = UInt64
        data.extend_from_slice(&(u64::MAX).to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("big_uint"),
            Some(&GGUFValue::UInt64(u64::MAX))
        );
    }

    #[test]
    fn test_parse_int64_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "big_int";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&11u32.to_le_bytes()); // value_type = Int64
        data.extend_from_slice(&(i64::MIN).to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(
            model.metadata.get("big_int"),
            Some(&GGUFValue::Int64(i64::MIN))
        );
    }

    #[test]
    fn test_parse_float64_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "double_val";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&12u32.to_le_bytes()); // value_type = Float64
        data.extend_from_slice(&1.125f64.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        if let Some(GGUFValue::Float64(val)) = model.metadata.get("double_val") {
            assert!((val - 1.125).abs() < 1e-10);
        } else {
            panic!("Expected Float64 value");
        }
    }

    #[test]
    fn test_parse_unsupported_value_type() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "unknown";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&99u32.to_le_bytes()); // Invalid value_type

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::UnsupportedOperation { .. }
        ));
    }

    #[test]
    fn test_parse_all_value_types() {
        // Test file with all supported value types
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&12u64.to_le_bytes()); // metadata_count = 12

        // UInt8
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"u8");
        data.extend_from_slice(&0u32.to_le_bytes());
        data.push(100u8);

        // Int8
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"i8");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-50i8).to_le_bytes());

        // UInt16
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"u16");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&1000u16.to_le_bytes());

        // Int16
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"i16");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&(-500i16).to_le_bytes());

        // UInt32
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"u32");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&100_000_u32.to_le_bytes());

        // Int32
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"i32");
        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&(-50000i32).to_le_bytes());

        // Float32
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"f32");
        data.extend_from_slice(&6u32.to_le_bytes());
        data.extend_from_slice(&1.5f32.to_le_bytes());

        // Bool
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(b"bool");
        data.extend_from_slice(&7u32.to_le_bytes());
        data.push(1u8);

        // String
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"str");
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(b"test");

        // UInt64
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"u64");
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&1_000_000u64.to_le_bytes());

        // Int64
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"i64");
        data.extend_from_slice(&11u32.to_le_bytes());
        data.extend_from_slice(&(-500_000_i64).to_le_bytes());

        // Float64
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"f64");
        data.extend_from_slice(&12u32.to_le_bytes());
        data.extend_from_slice(&2.5f64.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 12);
        assert_eq!(model.metadata.get("u8"), Some(&GGUFValue::UInt8(100)));
        assert_eq!(model.metadata.get("i8"), Some(&GGUFValue::Int8(-50)));
        assert_eq!(model.metadata.get("u16"), Some(&GGUFValue::UInt16(1000)));
        assert_eq!(model.metadata.get("i16"), Some(&GGUFValue::Int16(-500)));
        assert_eq!(model.metadata.get("u32"), Some(&GGUFValue::UInt32(100_000)));
        assert_eq!(model.metadata.get("i32"), Some(&GGUFValue::Int32(-50000)));
        assert_eq!(model.metadata.get("bool"), Some(&GGUFValue::Bool(true)));
        assert_eq!(
            model.metadata.get("str"),
            Some(&GGUFValue::String("test".to_string()))
        );
        assert_eq!(
            model.metadata.get("u64"),
            Some(&GGUFValue::UInt64(1_000_000))
        );
        assert_eq!(model.metadata.get("i64"), Some(&GGUFValue::Int64(-500_000)));

        // Check floats with tolerance
        if let Some(GGUFValue::Float32(val)) = model.metadata.get("f32") {
            assert!((val - 1.5).abs() < 1e-5);
        } else {
            panic!("Expected f32");
        }
        if let Some(GGUFValue::Float64(val)) = model.metadata.get("f64") {
            assert!((val - 2.5).abs() < 1e-10);
        } else {
            panic!("Expected f64");
        }
    }

    #[test]
    fn test_parse_array_uint32() {
        // GGUF header with 1 metadata item (Array of UInt32)
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata: key = "test.array", value_type = Array (9)
        let key = "test.array";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length
        data.extend_from_slice(key.as_bytes()); // key string
        data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
        data.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
        data.extend_from_slice(&3u64.to_le_bytes()); // array_len = 3
        data.extend_from_slice(&1u32.to_le_bytes()); // element 0
        data.extend_from_slice(&2u32.to_le_bytes()); // element 1
        data.extend_from_slice(&3u32.to_le_bytes()); // element 2

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 1);
        if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.array") {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], GGUFValue::UInt32(1));
            assert_eq!(arr[1], GGUFValue::UInt32(2));
            assert_eq!(arr[2], GGUFValue::UInt32(3));
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_parse_array_string() {
        // GGUF header with 1 metadata item (Array of strings)
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        // Metadata: key = "test.strings", value_type = Array (9)
        let key = "test.strings";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
        data.extend_from_slice(&8u32.to_le_bytes()); // element_type = String
        data.extend_from_slice(&2u64.to_le_bytes()); // array_len = 2

        // String element 0: "hello"
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");

        // String element 1: "world"
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"world");

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.metadata.len(), 1);
        if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.strings") {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], GGUFValue::String("hello".to_string()));
            assert_eq!(arr[1], GGUFValue::String("world".to_string()));
        } else {
            panic!("Expected Array value");
        }
    }

    #[test]
    fn test_parse_empty_array() {
        // GGUF header with empty array
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "empty";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
        data.extend_from_slice(&4u32.to_le_bytes()); // element_type = UInt32
        data.extend_from_slice(&0u64.to_le_bytes()); // array_len = 0

        let model = GGUFModel::from_bytes(&data).unwrap();
        if let Some(GGUFValue::Array(arr)) = model.metadata.get("empty") {
            assert_eq!(arr.len(), 0);
        } else {
            panic!("Expected empty Array");
        }
    }

    #[test]
    fn test_get_tensor_f32_unquantized() {
        // Create a GGUF file with F32 tensor
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version = 3
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor: name = "weights", dims = [2, 3], qtype = F32 (0)
        let tensor_name = "weights";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        data.extend_from_slice(&2u64.to_le_bytes()); // dim[0] = 2
        data.extend_from_slice(&3u64.to_le_bytes()); // dim[1] = 3
        data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes()); // qtype = F32

        // Tensor offset is 0 (relative to tensor data section start)
        data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Pad to 32-byte alignment
        while data.len() % GGUF_ALIGNMENT != 0 {
            data.push(0);
        }

        // Add F32 tensor data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for val in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&val.to_le_bytes());
        }

        let model = GGUFModel::from_bytes(&data).unwrap();
        let values = model.get_tensor_f32("weights", &data).unwrap();

        assert_eq!(values.len(), 6);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get_tensor_f32_not_found() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        let result = model.get_tensor_f32("nonexistent", &data);

        assert!(result.is_err());
        if let Err(RealizarError::UnsupportedOperation { reason, .. }) = result {
            assert!(reason.contains("not found"));
        }
    }

    #[test]
    fn test_get_tensor_f32_q4_0() {
        // Create a GGUF file with Q4_0 tensor
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor: name = "quant_weights", dims = [64] (2 blocks), qtype = Q4_0 (2)
        let tensor_name = "quant_weights";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&64u64.to_le_bytes()); // dim[0] = 64 (2 blocks of 32)
        data.extend_from_slice(&GGUF_TYPE_Q4_0.to_le_bytes());

        // Tensor offset is 0 (relative to tensor data section start)
        data.extend_from_slice(&0u64.to_le_bytes());

        // Pad to 32-byte alignment
        while data.len() % GGUF_ALIGNMENT != 0 {
            data.push(0);
        }

        // Add Q4_0 data: 2 blocks (20 bytes each)
        // Block 1: scale = 1.0, quants = 16 bytes
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&[0x10; 16]); // 4-bit values

        // Block 2: scale = 2.0, quants = 16 bytes
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&[0x21; 16]);

        let model = GGUFModel::from_bytes(&data).unwrap();
        let values = model.get_tensor_f32("quant_weights", &data).unwrap();

        // Verify size is correct
        assert_eq!(values.len(), 64);

        // Values should be dequantized (non-zero)
        assert!(values.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_get_tensor_f32_q8_0() {
        // Create a GGUF file with Q8_0 tensor
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor: dims = [32] (1 block), qtype = Q8_0 (8)
        let tensor_name = "q8_weights";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&32u64.to_le_bytes()); // dim[0] = 32 (1 block)
        data.extend_from_slice(&GGUF_TYPE_Q8_0.to_le_bytes());

        // Tensor offset is 0 (relative to tensor data section start)
        data.extend_from_slice(&0u64.to_le_bytes());

        // Pad to 32-byte alignment
        while data.len() % GGUF_ALIGNMENT != 0 {
            data.push(0);
        }

        // Add Q8_0 data: 1 block (36 bytes: 4 for scale + 32 for quants)
        data.extend_from_slice(&0.5f32.to_le_bytes());
        for i in 0i32..32 {
            // Test data uses i8 range [0, 31] - safe to convert
            data.push(u8::try_from(i).unwrap());
        }

        let model = GGUFModel::from_bytes(&data).unwrap();
        let values = model.get_tensor_f32("q8_weights", &data).unwrap();

        assert_eq!(values.len(), 32);
        // First value should be approximately 0.5 * 0 = 0.0
        assert!((values[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_qtype() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor with unsupported qtype
        let tensor_name = "bad_tensor";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(&999u32.to_le_bytes()); // Invalid qtype

        // Calculate offset
        let tensor_offset = (data.len() + 8) as u64;
        data.extend_from_slice(&tensor_offset.to_le_bytes());

        let model = GGUFModel::from_bytes(&data).unwrap();
        let result = model.get_tensor_f32("bad_tensor", &data);

        assert!(result.is_err());
        if let Err(RealizarError::UnsupportedOperation { reason, .. }) = result {
            assert!(reason.contains("Unsupported quantization type"));
        }
    }

    // ============================================================
    // QuantizedGGUFTransformer::generate() tests
    // Per benchmark-model-runners-spec.md "What's Remaining" item 1
    // ============================================================

    #[test]
    fn test_generate_config_default() {
        let config = QuantizedGenerateConfig::default();
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generate_config_builder() {
        let config = QuantizedGenerateConfig::default()
            .with_max_tokens(128)
            .with_temperature(0.7)
            .with_top_k(40)
            .with_stop_tokens(vec![50256]);

        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.stop_tokens, vec![50256]);
    }

    #[test]
    fn test_generate_config_deterministic() {
        // Temperature 0.0 = greedy decoding
        let config = QuantizedGenerateConfig::deterministic(32);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
        assert_eq!(config.max_tokens, 32);
    }
}
