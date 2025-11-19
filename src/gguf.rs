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
    io::{Cursor, Read},
};

use crate::error::{RealizarError, Result};

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF versions
pub const GGUF_VERSION_V3: u32 = 3;

/// GGUF quantization type: F32 (unquantized float32)
pub const GGUF_TYPE_F32: u32 = 0;

/// GGUF quantization type: `Q4_0` (4-bit quantization, block size 32)
pub const GGUF_TYPE_Q4_0: u32 = 2;

/// GGUF quantization type: `Q8_0` (8-bit quantization, block size 32)
pub const GGUF_TYPE_Q8_0: u32 = 8;

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

/// GGUF model container
#[derive(Debug, Clone)]
pub struct GGUFModel {
    /// File header
    pub header: GGUFHeader,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
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

        Ok(Self {
            header,
            metadata,
            tensors,
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

        // Convert offset to usize
        let offset =
            usize::try_from(tensor.offset).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_offset".to_string(),
                reason: format!("Offset {} exceeds platform usize limit", tensor.offset),
            })?;

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
                // Q8_0 quantized data
                use crate::quantize::dequantize_q8_0;

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
                let mut values = dequantize_q8_0(bytes)?;

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

        // Calculate offset: current position + 8 bytes for offset field
        let tensor_offset = (data.len() + 8) as u64;
        data.extend_from_slice(&tensor_offset.to_le_bytes()); // offset

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

        // Calculate offset
        let tensor_offset = (data.len() + 8) as u64;
        data.extend_from_slice(&tensor_offset.to_le_bytes());

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

        // Calculate offset
        let tensor_offset = (data.len() + 8) as u64;
        data.extend_from_slice(&tensor_offset.to_le_bytes());

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
}
