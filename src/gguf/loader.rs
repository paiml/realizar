//! GGUF model loading and parsing
//!
//! Contains GGUFModel and GGUFTransformer parsing implementations extracted from monolith.
//! This handles the binary format parsing and tensor info extraction.

use crate::error::{RealizarError, Result};
use crate::gguf::utils::gpt2_unicode_to_byte;
use crate::gguf::{
    GGUFConfig, GGUFHeader, GGUFModel, GGUFTransformer, GGUFTransformerLayer, GGUFValue,
    TensorInfo, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q2_K,
    GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};
use std::collections::HashMap;
use std::io::{Cursor, Read};

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
            // GGUF stores dimensions in GGML order (reversed from standard row-major)
            // We need to reverse them to get the correct shape [out_dim, in_dim]
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(cursor)?);
            }
            dims.reverse();

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

                // Q4_0 block: 32 elements
                // Layout: 1×f16 scale (2 bytes) + 16 bytes (32×4-bit values) = 18 bytes
                const BLOCK_BYTES: usize = 18;
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

                // Q8_0 block size: 34 bytes (2 for f16 scale + 32 for quants)
                const BLOCK_BYTES: usize = 34;
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
            GGUF_TYPE_Q2_K => {
                // Q2_K quantized data (K-quantization) - 2 bits per weight
                use crate::quantize::{dequantize_q2_k, QK_K};

                // Q2_K super-block size: 84 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 84;

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
                let mut values = dequantize_q2_k(bytes)?;

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

    /// Get number of key-value heads from metadata (for GQA)
    pub fn num_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.head_count_kv", arch);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get RoPE frequency base from metadata
    /// Different models use different bases (LLaMA: 10000, Qwen2: 1000000)
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{}.rope.freq_base", arch);
        if let Some(GGUFValue::Float32(base)) = self.metadata.get(&key) {
            Some(*base)
        } else {
            None
        }
    }

    /// Get RMSNorm epsilon from metadata
    /// Different models use different values (LLaMA: 1e-5, Qwen2: 1e-6)
    pub fn rms_epsilon(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.layer_norm_rms_epsilon", arch);
        if let Some(GGUFValue::Float32(eps)) = self.metadata.get(&key) {
            Some(*eps)
        } else {
            None
        }
    }

    /// Get RoPE type from metadata or infer from architecture
    /// Returns: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    /// Per llama.cpp: LLAMA_ROPE_TYPE_NORM = 0, LLAMA_ROPE_TYPE_NEOX = 2
    ///
    /// Architecture-based inference matches llama.cpp's llama-model.cpp:7763-7811
    pub fn rope_type(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{}.rope.scaling.type", arch);
        // Try rope type from scaling type first
        if let Some(GGUFValue::String(s)) = self.metadata.get(&key) {
            match s.as_str() {
                "none" | "linear" => return Some(0), // NORM style
                "yarn" | "neox" => return Some(2),   // NEOX style
                _ => {},
            }
        }
        // Infer rope type from architecture (matches llama.cpp llama-model.cpp:7763-7811)
        // NEOX style (type 2): pairs offset by n_rot/2
        let arch_lower = arch.to_lowercase();
        let neox_architectures = [
            "qwen",
            "qwen2",
            "qwen3",
            "stablelm",
            "phi2",
            "phi3",
            "gemma",
            "gemma2",
            "gemma3",
            "starcoder2",
            "gptneox",
            "falcon",
            "codeshell",
            "orion",
            "bert",
            "nomic-bert",
            "dbrx",
            "olmo2",
            "olmoe",
            "plamo",
            "plamo2",
            "openelm",
            "exaone",
            "minicpm3",
            "nemotron",
            "internlm2",
            "deepseek2",
        ];
        for neox_arch in neox_architectures {
            if arch_lower.contains(neox_arch) {
                return Some(2); // NEOX style
            }
        }
        // NORM style (type 0): adjacent pairs - default for LLaMA, TinyLlama
        Some(0)
    }

    /// Get BOS (beginning of sentence) token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get EOS (end of sentence) token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get vocabulary tokens from metadata
    ///
    /// Returns the token strings indexed by token ID.
    /// Uses "tokenizer.ggml.tokens" key from GGUF metadata.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        if let Some(GGUFValue::Array(arr)) = self.metadata.get("tokenizer.ggml.tokens") {
            let tokens: Vec<String> = arr
                .iter()
                .filter_map(|v| {
                    if let GGUFValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            if tokens.is_empty() {
                None
            } else {
                Some(tokens)
            }
        } else {
            None
        }
    }

    /// Decode token IDs to text using vocabulary
    ///
    /// Returns decoded string. Unknown tokens are replaced with "�".
    /// Handles BPE markers:
    /// - GPT-2 style: Ġ (U+0120) → space, Ċ (U+010A) → newline
    /// - SentencePiece: ▁ (U+2581) → space
    /// - Byte tokens: <0xHH> → actual byte value
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        if let Some(vocab) = self.vocabulary() {
            // Detect tokenizer type from metadata
            let is_gpt2_style = self
                .metadata
                .get("tokenizer.ggml.model")
                .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2"));

            // Collect raw tokens and convert byte tokens to actual bytes
            let mut bytes: Vec<u8> = Vec::new();

            for &id in token_ids {
                let token = vocab
                    .get(id as usize)
                    .map_or("�", std::string::String::as_str);

                // Check if this is a byte token like <0xE6>
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    if let Ok(byte_val) = u8::from_str_radix(&token[3..5], 16) {
                        bytes.push(byte_val);
                        continue;
                    }
                }

                // For GPT-2 style tokenizers, decode byte-level BPE properly
                // Each unicode character in the token represents a raw byte
                if is_gpt2_style {
                    for c in token.chars() {
                        if let Some(byte) = gpt2_unicode_to_byte(c) {
                            bytes.push(byte);
                        }
                    }
                } else {
                    // SentencePiece style - tokens are regular strings
                    bytes.extend_from_slice(token.as_bytes());
                }
            }

            // Decode bytes as UTF-8 (lossy for invalid sequences)
            let raw = String::from_utf8_lossy(&bytes).into_owned();

            // Post-process BPE markers (only for SentencePiece, GPT-2 already handled)
            if !is_gpt2_style {
                raw.replace('▁', " ") // SentencePiece word boundary
            } else {
                raw
            }
        } else {
            // Fallback to ASCII if no vocabulary
            token_ids
                .iter()
                .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
                .collect()
        }
    }

    /// Encode text to token IDs using vocabulary
    ///
    /// Uses greedy longest-match tokenization with special token priority.
    /// Returns None if no vocabulary is available.
    ///
    /// Supports both tokenizer types:
    /// - SentencePiece (llama): Uses `▁` (U+2581) for word boundaries
    /// - GPT-2 (qwen2, gpt2): Uses `Ġ` (U+0120) for space prefixes
    #[must_use]
    pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
        let vocab = self.vocabulary()?;

        // Build reverse lookup: token string -> token ID
        let token_to_id: std::collections::HashMap<&str, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id as u32))
            .collect();

        // Identify special tokens (high-ID tokens with <|...|> pattern)
        // These need priority matching to avoid being split by greedy algorithm
        let special_tokens: Vec<(&str, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(id, tok)| *id >= 151643 && tok.starts_with("<|") && tok.ends_with("|>"))
            .map(|(id, tok)| (tok.as_str(), id as u32))
            .collect();

        // Detect tokenizer type from metadata
        // GPT-2 style uses Ġ (U+0120), SentencePiece uses ▁ (U+2581)
        let is_gpt2_style = self
            .metadata
            .get("tokenizer.ggml.model")
            .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2"));

        let space_char = if is_gpt2_style { '\u{0120}' } else { '▁' };

        // Split text on special tokens first, preserving them
        let mut segments: Vec<(bool, &str)> = Vec::new(); // (is_special, text)
        let mut text_remaining = text;
        while !text_remaining.is_empty() {
            // Find earliest special token match
            let mut earliest_match: Option<(usize, &str, u32)> = None;
            for &(special_tok, special_id) in &special_tokens {
                if let Some(pos) = text_remaining.find(special_tok) {
                    if earliest_match.is_none()
                        || pos < earliest_match.as_ref().map_or(usize::MAX, |m| m.0)
                    {
                        earliest_match = Some((pos, special_tok, special_id));
                    }
                }
            }

            if let Some((pos, special_tok, _)) = earliest_match {
                if pos > 0 {
                    segments.push((false, &text_remaining[..pos]));
                }
                segments.push((true, special_tok));
                text_remaining = &text_remaining[pos + special_tok.len()..];
            } else {
                segments.push((false, text_remaining));
                break;
            }
        }

        let mut tokens = Vec::new();

        for (is_special, segment) in segments {
            if is_special {
                // Direct lookup for special token
                if let Some(&id) = token_to_id.get(segment) {
                    tokens.push(id);
                }
                continue;
            }

            // Process non-special segment with character replacement
            let text_with_prefix = if is_gpt2_style {
                segment.to_string()
            } else if segment.starts_with(' ') {
                segment.to_string()
            } else {
                format!(" {}", segment)
            };

            let processed = if is_gpt2_style {
                text_with_prefix
                    .replace(' ', &space_char.to_string())
                    .replace('\n', "\u{010A}") // Ċ = GPT-2 newline
            } else {
                text_with_prefix.replace(' ', &space_char.to_string())
            };

            let mut remaining = processed.as_str();

            while !remaining.is_empty() {
                // Greedy longest match using character boundaries (not byte indices)
                let mut best_byte_len = 0;
                let mut best_id = None;

                // Collect character byte offsets for proper slicing
                let char_indices: Vec<usize> = remaining
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(remaining.len()))
                    .collect();

                // Try all prefixes up to 32 chars (reasonable max token length)
                for char_count in 1..=char_indices.len().saturating_sub(1).min(32) {
                    let byte_end = char_indices[char_count];
                    let prefix = &remaining[..byte_end];
                    if let Some(&id) = token_to_id.get(prefix) {
                        best_byte_len = byte_end;
                        best_id = Some(id);
                    }
                }

                if let Some(id) = best_id {
                    tokens.push(id);
                    remaining = &remaining[best_byte_len..];
                } else {
                    // No match found - try single UTF-8 char as byte tokens
                    // SAFETY: remaining is non-empty (loop condition guarantees this)
                    let ch = remaining
                        .chars()
                        .next()
                        .expect("loop invariant: remaining non-empty");
                    let ch_len = ch.len_utf8();

                    // Look for byte tokens like <0x48> for 'H'
                    for byte in remaining[..ch_len].bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = token_to_id.get(byte_token.as_str()) {
                            tokens.push(id);
                        } else {
                            // Unknown byte - use a common unknown token ID (usually 0 or 1)
                            tokens.push(0);
                        }
                    }
                    remaining = &remaining[ch_len..];
                }
            }
        }

        Some(tokens)
    }
}

use crate::gguf::{
    OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
    QuantizedGGUFTransformer,
};

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

        // Load output norm (raw gamma values - no delta transformation needed)
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", file_data)?;
        let output_norm_bias = model.get_tensor_f32("output_norm.bias", file_data).ok();

        // Load LM head (output projection)
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = model
            .get_tensor_f32("output.weight", file_data)
            .or_else(|_| model.get_tensor_f32("token_embd.weight", file_data))?;
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

        // Attention norm weights
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
}

impl OwnedQuantizedModel {
    /// Create owned model from memory-mapped GGUF file
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    pub fn from_mapped(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        let data = mapped.data();
        let transformer = QuantizedGGUFTransformer::from_gguf(&mapped.model, data)?;

        // Get config for dimension calculations
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // Convert layers to owned (passing config for dimensions)
        let layers: Vec<OwnedQuantizedLayer> = transformer
            .layers
            .iter()
            .map(|l| OwnedQuantizedLayer::from_borrowed(l, data, config))
            .collect();

        Ok(Self {
            config: transformer.config.clone(),
            token_embedding: transformer.token_embedding,
            layers,
            output_norm_weight: transformer.output_norm_weight,
            output_norm_bias: transformer.output_norm_bias,
            // LM head: [hidden_dim] -> [vocab_size]
            lm_head_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &transformer.lm_head_weight,
                data,
                hidden_dim,
                vocab_size,
            ),
            lm_head_bias: transformer.lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Create a model for testing purposes
    ///
    /// This constructor handles the internal CUDA fields automatically,
    /// allowing external tests to construct models without accessing pub(crate) fields.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `token_embedding` - Token embedding weights
    /// * `layers` - Quantized transformer layers
    /// * `output_norm_weight` - Output normalization weight
    /// * `output_norm_bias` - Optional output normalization bias
    /// * `lm_head_weight` - Language model head weight
    /// * `lm_head_bias` - Optional language model head bias
    #[must_use]
    pub fn new_for_test(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<OwnedQuantizedLayer>,
        output_norm_weight: Vec<f32>,
        output_norm_bias: Option<Vec<f32>>,
        lm_head_weight: OwnedQuantizedTensor,
        lm_head_bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Create model from memory-mapped APR file (SHOWCASE-APR-GPU)
    ///
    /// Converts APR Q4K format to GGUF-compatible model for GPU inference.
    /// The raw Q4K tensor data is byte-compatible between formats.
    ///
    /// # Arguments
    /// * `apr` - Memory-mapped APR model
    ///
    /// # Errors
    /// Returns error if APR format is invalid or missing required tensors.
    pub fn from_apr(apr: &crate::apr::MappedAprModel) -> Result<Self> {
        use crate::apr::MappedAprModel;

        let data = apr.data();
        let data_offset = apr.data_offset() as usize;

        // Build config from APR metadata
        let hidden_dim = apr.metadata.hidden_size.unwrap_or(1536);
        let num_layers = apr.metadata.num_layers.unwrap_or(28);
        let num_heads = apr.metadata.num_heads.unwrap_or(12);
        let num_kv_heads = apr.metadata.num_kv_heads.unwrap_or(2);
        let intermediate_dim = apr.metadata.intermediate_size.unwrap_or(8960);
        let eps = apr.metadata.rms_norm_eps.unwrap_or(1e-6);
        let rope_theta = apr.metadata.rope_theta.unwrap_or(1_000_000.0);

        // Infer vocab_size from embedding tensor if metadata is 0 or missing
        let vocab_size = match apr.metadata.vocab_size {
            Some(v) if v > 0 => v,
            _ => {
                // Try to infer from embedding tensor shape
                apr.tensors
                    .iter()
                    .find(|t| {
                        t.name.contains("embed_tokens")
                            || t.name.contains("tok_embeddings")
                            || t.name.contains("token_embd")
                    })
                    .and_then(|t| t.shape.first().copied())
                    .unwrap_or(151936)
            },
        };

        let config = GGUFConfig {
            architecture: apr
                .metadata
                .architecture
                .clone()
                .unwrap_or_else(|| "qwen2".to_string()),
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_dim,
            eps,
            rope_theta,
            rope_type: 2, // NEOX style for Qwen2.5
            context_length: 32768,
        };

        // Helper to get tensor data
        let get_tensor = |name: &str| -> Result<&[u8]> {
            let tensor = apr
                .find_tensor(name)
                .ok_or_else(|| RealizarError::FormatError {
                    reason: format!("APR: tensor not found: {name}"),
                })?;
            let start = data_offset + tensor.offset as usize;
            let end = start + tensor.size as usize;
            if end > data.len() {
                return Err(RealizarError::FormatError {
                    reason: format!("APR: tensor {name} extends past EOF"),
                });
            }
            Ok(&data[start..end])
        };

        // Helper to get tensor qtype
        let get_qtype = |name: &str| -> u32 {
            apr.find_tensor(name)
                .map_or(0, |t| MappedAprModel::dtype_to_qtype(&t.dtype))
        };

        // Helper to make OwnedQuantizedTensor
        let make_tensor =
            |name: &str, in_dim: usize, out_dim: usize| -> Result<OwnedQuantizedTensor> {
                let tensor_data = get_tensor(name)?;
                let qtype = get_qtype(name);
                Ok(OwnedQuantizedTensor {
                    data: tensor_data.to_vec(),
                    in_dim,
                    out_dim,
                    qtype,
                })
            };

        // Load token embeddings (F32)
        let embed_name = apr
            .tensors
            .iter()
            .find(|t| {
                t.name.contains("embed_tokens")
                    || t.name.contains("tok_embeddings")
                    || t.name.contains("token_embd")
            })
            .map(|t| t.name.as_str())
            .ok_or_else(|| RealizarError::FormatError {
                reason: "APR: embedding tensor not found".to_string(),
            })?;

        let embed_data = get_tensor(embed_name)?;
        let embed_dtype = apr.find_tensor(embed_name).map(|t| t.dtype.as_str());
        let token_embedding: Vec<f32> = match embed_dtype {
            Some("F32") => embed_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Some("Q4_K") => {
                // Dequantize Q4_K embeddings
                crate::quantize::dequantize_q4_k(embed_data)?
            },
            Some(dtype) => {
                return Err(RealizarError::FormatError {
                    reason: format!("APR: unsupported embedding dtype: {dtype}"),
                });
            },
            None => {
                return Err(RealizarError::FormatError {
                    reason: "APR: embedding tensor dtype not found".to_string(),
                });
            },
        };

        // Build layers
        let mut layers = Vec::with_capacity(num_layers);
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        for layer_idx in 0..num_layers {
            // Find layer tensors (try multiple naming conventions)
            let q_name = format!("blk.{layer_idx}.attn_q.weight");
            let k_name = format!("blk.{layer_idx}.attn_k.weight");
            let v_name = format!("blk.{layer_idx}.attn_v.weight");
            let o_name = format!("blk.{layer_idx}.attn_output.weight");

            let gate_name = format!("blk.{layer_idx}.ffn_gate.weight");
            let up_name = format!("blk.{layer_idx}.ffn_up.weight");
            let down_name = format!("blk.{layer_idx}.ffn_down.weight");

            let attn_norm_name = format!("blk.{layer_idx}.attn_norm.weight");
            let ffn_norm_name = format!("blk.{layer_idx}.ffn_norm.weight");

            // Q/K/V weights
            let q_weight = make_tensor(&q_name, hidden_dim, hidden_dim)?;
            let k_weight = make_tensor(&k_name, hidden_dim, kv_dim)?;
            let v_weight = make_tensor(&v_name, hidden_dim, kv_dim)?;

            let qkv_weight = OwnedQKVWeights::Separate {
                q: q_weight,
                k: k_weight,
                v: v_weight,
            };

            // O projection
            let o_weight = make_tensor(&o_name, hidden_dim, hidden_dim)?;

            // FFN weights
            let ffn_gate_weight = make_tensor(&gate_name, hidden_dim, intermediate_dim)?;
            let ffn_up_weight = make_tensor(&up_name, hidden_dim, intermediate_dim)?;
            let ffn_down_weight = make_tensor(&down_name, intermediate_dim, hidden_dim)?;

            // Norm weights (F32)
            let attn_norm_data = get_tensor(&attn_norm_name)?;
            let ffn_norm_data = get_tensor(&ffn_norm_name)?;

            let attn_norm_weight: Vec<f32> = attn_norm_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let ffn_norm_weight: Vec<f32> = ffn_norm_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            layers.push(OwnedQuantizedLayer {
                attn_norm_weight,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias: None,
                attn_output_weight: o_weight,
                attn_output_bias: None,
                ffn_norm_weight: Some(ffn_norm_weight),
                ffn_norm_bias: None,
                ffn_gate_weight: Some(ffn_gate_weight),
                ffn_gate_bias: None,
                ffn_up_weight,
                ffn_up_bias: None,
                ffn_down_weight,
                ffn_down_bias: None,
            });
        }

        // Output norm
        let output_norm_name = apr
            .tensors
            .iter()
            .find(|t| t.name.contains("output_norm") || t.name.contains("norm.weight"))
            .map_or("output_norm.weight", |t| t.name.as_str());

        let output_norm_data = get_tensor(output_norm_name)?;
        let output_norm_weight: Vec<f32> = output_norm_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // LM head - prioritize exact match, then contains (excluding layer tensors)
        let lm_head_name = apr
            .tensors
            .iter()
            .find(|t| t.name == "output.weight" || t.name == "lm_head.weight")
            .or_else(|| {
                apr.tensors.iter().find(|t| {
                    !t.name.starts_with("blk.")
                        && (t.name.contains("output.weight") || t.name.contains("lm_head"))
                })
            })
            .map_or("output.weight", |t| t.name.as_str());

        let lm_head_weight = make_tensor(lm_head_name, hidden_dim, vocab_size)?;

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Serialize model to APR format with quantized weights preserved
    ///
    /// Creates a valid .apr file that can be loaded via `from_apr()`.
    /// Quantization types (Q4_K, Q6_K, etc.) are preserved in the tensor dtypes.
    ///
    /// # Returns
    ///
    /// Raw bytes in APR v2 format
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_apr_bytes(&self) -> Result<Vec<u8>> {
        use crate::apr::{ALIGNMENT, HEADER_SIZE, MAGIC};

        // Helper to convert GGML qtype to APR dtype
        fn qtype_to_dtype(qtype: u32) -> &'static str {
            match qtype {
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
                _ => "F32",
            }
        }

        // Helper to convert dtype string to byte for binary tensor entry
        // GH-191 FIX: Use GGML dtype values directly so they match TensorEntry::from_binary reader.
        // The reader maps these bytes back to dtype strings; mismatched values caused
        // all quantized tensors to silently fall through to F32.
        fn dtype_to_byte(dtype: &str) -> u8 {
            match dtype {
                "F32" => 0,
                "F16" => 1,
                "BF16" => 30,   // GGML BF16 type
                "Q4_0" => 2,    // GGML type 2
                "Q4_1" => 3,    // GGML type 3
                "Q5_0" => 6,    // GGML type 6
                "Q5_1" => 7,    // GGML type 7
                "Q8_0" => 8,    // GGML type 8
                "Q8_1" => 9,    // GGML type 9
                "Q2_K" => 10,   // GGML type 10
                "Q3_K" => 11,   // GGML type 11
                "Q4_K" => 12,   // GGML type 12
                "Q5_K" => 13,   // GGML type 13
                "Q6_K" => 14,   // GGML type 14
                "IQ2_XXS" => 16, // GGML type 16
                "IQ2_XS" => 17,  // GGML type 17
                _ => {
                    eprintln!("WARN: Unknown dtype '{}' in dtype_to_byte, writing as F32", dtype);
                    0
                }
            }
        }

        // Helper to write tensor entry to binary format
        fn write_tensor_entry(
            name: &str,
            dtype: &str,
            shape: &[usize],
            offset: u64,
            size: u64,
        ) -> Vec<u8> {
            let mut entry = Vec::new();

            // Name: 2-byte length + bytes
            let name_bytes = name.as_bytes();
            entry.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            entry.extend_from_slice(name_bytes);

            // Dtype: 1 byte
            entry.push(dtype_to_byte(dtype));

            // Shape: 1-byte ndim + 8-byte dims
            entry.push(shape.len() as u8);
            for &dim in shape {
                entry.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // Offset and size: 8 bytes each
            entry.extend_from_slice(&offset.to_le_bytes());
            entry.extend_from_slice(&size.to_le_bytes());

            entry
        }

        // Collect all tensors
        struct TensorInfo {
            name: String,
            dtype: String,
            shape: Vec<usize>,
            data: Vec<u8>,
        }

        let mut tensors: Vec<TensorInfo> = Vec::new();

        // Token embedding (F32)
        let embed_bytes: Vec<u8> = self
            .token_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.push(TensorInfo {
            name: "token_embd.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![self.config.vocab_size, self.config.hidden_dim],
            data: embed_bytes,
        });

        // Layers
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let kv_dim = self.config.num_kv_heads * head_dim;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention norm (F32)
            let norm_bytes: Vec<u8> = layer
                .attn_norm_weight
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            tensors.push(TensorInfo {
                name: format!("blk.{layer_idx}.attn_norm.weight"),
                dtype: "F32".to_string(),
                shape: vec![self.config.hidden_dim],
                data: norm_bytes,
            });

            // QKV weights (quantized)
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    tensors.push(TensorInfo {
                        name: format!("blk.{layer_idx}.attn_q.weight"),
                        dtype: qtype_to_dtype(q.qtype).to_string(),
                        shape: vec![self.config.hidden_dim, self.config.hidden_dim],
                        data: q.data.clone(),
                    });
                    tensors.push(TensorInfo {
                        name: format!("blk.{layer_idx}.attn_k.weight"),
                        dtype: qtype_to_dtype(k.qtype).to_string(),
                        shape: vec![kv_dim, self.config.hidden_dim],
                        data: k.data.clone(),
                    });
                    tensors.push(TensorInfo {
                        name: format!("blk.{layer_idx}.attn_v.weight"),
                        dtype: qtype_to_dtype(v.qtype).to_string(),
                        shape: vec![kv_dim, self.config.hidden_dim],
                        data: v.data.clone(),
                    });
                },
                OwnedQKVWeights::Fused(t) => {
                    // Store as fused QKV tensor
                    tensors.push(TensorInfo {
                        name: format!("blk.{layer_idx}.attn_qkv.weight"),
                        dtype: qtype_to_dtype(t.qtype).to_string(),
                        shape: vec![t.out_dim, t.in_dim],
                        data: t.data.clone(),
                    });
                },
            }

            // Output projection (quantized)
            tensors.push(TensorInfo {
                name: format!("blk.{layer_idx}.attn_output.weight"),
                dtype: qtype_to_dtype(layer.attn_output_weight.qtype).to_string(),
                shape: vec![self.config.hidden_dim, self.config.hidden_dim],
                data: layer.attn_output_weight.data.clone(),
            });

            // FFN norm (F32)
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                let norm_bytes: Vec<u8> = ffn_norm.iter().flat_map(|f| f.to_le_bytes()).collect();
                tensors.push(TensorInfo {
                    name: format!("blk.{layer_idx}.ffn_norm.weight"),
                    dtype: "F32".to_string(),
                    shape: vec![self.config.hidden_dim],
                    data: norm_bytes,
                });
            }

            // FFN weights (quantized)
            if let Some(ref gate) = layer.ffn_gate_weight {
                tensors.push(TensorInfo {
                    name: format!("blk.{layer_idx}.ffn_gate.weight"),
                    dtype: qtype_to_dtype(gate.qtype).to_string(),
                    shape: vec![self.config.intermediate_dim, self.config.hidden_dim],
                    data: gate.data.clone(),
                });
            }

            tensors.push(TensorInfo {
                name: format!("blk.{layer_idx}.ffn_up.weight"),
                dtype: qtype_to_dtype(layer.ffn_up_weight.qtype).to_string(),
                shape: vec![self.config.intermediate_dim, self.config.hidden_dim],
                data: layer.ffn_up_weight.data.clone(),
            });

            tensors.push(TensorInfo {
                name: format!("blk.{layer_idx}.ffn_down.weight"),
                dtype: qtype_to_dtype(layer.ffn_down_weight.qtype).to_string(),
                shape: vec![self.config.hidden_dim, self.config.intermediate_dim],
                data: layer.ffn_down_weight.data.clone(),
            });
        }

        // Output norm (F32)
        let output_norm_bytes: Vec<u8> = self
            .output_norm_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.push(TensorInfo {
            name: "output_norm.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![self.config.hidden_dim],
            data: output_norm_bytes,
        });

        // LM head (quantized)
        tensors.push(TensorInfo {
            name: "output.weight".to_string(),
            dtype: qtype_to_dtype(self.lm_head_weight.qtype).to_string(),
            shape: vec![self.config.vocab_size, self.config.hidden_dim],
            data: self.lm_head_weight.data.clone(),
        });

        // Build metadata JSON
        let metadata = serde_json::json!({
            "model_type": "transformer_lm",
            "architecture": self.config.architecture,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "intermediate_size": self.config.intermediate_dim,
            "rms_norm_eps": self.config.eps,
            "rope_theta": self.config.rope_theta,
            "context_length": self.config.context_length,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize metadata: {e}"),
            })?;
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Build tensor index and data
        let mut tensor_index_bytes: Vec<u8> = Vec::new();
        let mut tensor_data_bytes: Vec<u8> = Vec::new();

        for tensor in &tensors {
            // Align tensor data to 64 bytes
            let padding = (ALIGNMENT - (tensor_data_bytes.len() % ALIGNMENT)) % ALIGNMENT;
            tensor_data_bytes.extend(std::iter::repeat_n(0u8, padding));

            let offset = tensor_data_bytes.len() as u64;
            let size = tensor.data.len() as u64;

            tensor_index_bytes.extend(write_tensor_entry(
                &tensor.name,
                &tensor.dtype,
                &tensor.shape,
                offset,
                size,
            ));

            tensor_data_bytes.extend_from_slice(&tensor.data);
        }

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

        // Build header
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags (quantized = bit 0)
        header[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset.to_le_bytes());
        // checksum at 40-43 (leave as 0 for now)

        // Combine all parts
        let total_size =
            HEADER_SIZE + metadata_padded_len + tensor_index_bytes.len() + tensor_data_bytes.len();
        let mut result = Vec::with_capacity(total_size);
        result.extend_from_slice(&header);
        result.extend_from_slice(&metadata_bytes);
        result.resize(HEADER_SIZE + metadata_padded_len, 0); // pad metadata
        result.extend_from_slice(&tensor_index_bytes);
        result.extend_from_slice(&tensor_data_bytes);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};

    // ============================================================================
    // GGUFModel parsing tests
    // ============================================================================

    #[test]
    fn test_gguf_model_from_bytes_llama() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok(), "Failed to parse GGUF: {:?}", model.err());

        let model = model.unwrap();
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, GGUF_VERSION_V3);
    }

    #[test]
    fn test_gguf_model_from_bytes_phi2() {
        let data = build_minimal_phi2_gguf(100, 64, 256, 4);
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok(), "Failed to parse GGUF: {:?}", model.err());
    }

    #[test]
    fn test_gguf_model_tensors_parsed() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        // Should have multiple tensors (embedding, layer weights, etc.)
        assert!(!model.tensors.is_empty());
    }

    #[test]
    fn test_gguf_model_metadata_parsed() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        // Should have metadata (architecture, hidden_dim, etc.)
        assert!(!model.metadata.is_empty());
    }

    #[test]
    fn test_gguf_model_tensor_data_start() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        // tensor_data_start should be 32-byte aligned
        assert!(model.tensor_data_start % GGUF_ALIGNMENT == 0);
    }

    #[test]
    fn test_gguf_model_invalid_magic() {
        let mut data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        // Corrupt the magic number
        data[0] = 0xFF;
        data[1] = 0xFF;
        data[2] = 0xFF;
        data[3] = 0xFF;

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_architecture() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let arch = model.architecture();
        assert!(arch.is_some());
    }

    #[test]
    fn test_gguf_model_embedding_dim() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let dim = model.embedding_dim();
        assert!(dim.is_some());
        assert_eq!(dim.unwrap(), 64);
    }

    #[test]
    fn test_gguf_model_num_layers() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let layers = model.num_layers();
        assert!(layers.is_some());
    }

    #[test]
    fn test_gguf_model_num_heads() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let heads = model.num_heads();
        assert!(heads.is_some());
        assert_eq!(heads.unwrap(), 4);
    }

    #[test]
    fn test_gguf_model_num_kv_heads() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 2); // num_kv_heads=2
        let model = GGUFModel::from_bytes(&data).unwrap();

        let kv_heads = model.num_kv_heads();
        assert!(kv_heads.is_some());
    }

    #[test]
    fn test_gguf_model_context_length() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let ctx = model.context_length();
        // May or may not be present depending on test factory
        assert!(ctx.is_none() || ctx.unwrap() > 0);
    }

    #[test]
    fn test_gguf_model_rope_freq_base() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        // May have rope_freq_base metadata
        let _theta = model.rope_freq_base();
    }

    #[test]
    fn test_gguf_model_get_tensor_f32() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        // Try to get token embedding (should be F32)
        let emb = model.get_tensor_f32("token_embd.weight", &data);
        assert!(emb.is_ok());
        let emb = emb.unwrap();
        assert!(!emb.is_empty());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_not_found() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();

        let result = model.get_tensor_f32("nonexistent_tensor", &data);
        assert!(result.is_err());
    }

    // ============================================================================
    // GGUFTransformer tests
    // ============================================================================

    #[test]
    fn test_gguf_transformer_from_mapped() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data);

        assert!(transformer.is_ok(), "Failed: {:?}", transformer.err());
    }

    #[test]
    fn test_gguf_transformer_config() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.config.num_heads, 4);
    }

    #[test]
    fn test_gguf_transformer_layers() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        // Should have at least 1 layer
        assert!(!transformer.layers.is_empty());
    }

    #[test]
    fn test_gguf_transformer_token_embedding() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        // token_embedding should be vocab_size * hidden_dim
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
    }

    #[test]
    fn test_gguf_transformer_output_norm() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        // output_norm_weight should be hidden_dim
        assert_eq!(transformer.output_norm_weight.len(), 64);
    }

    #[test]
    fn test_gguf_transformer_lm_head() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        // lm_head should be vocab_size * hidden_dim
        assert!(!transformer.lm_head_weight.is_empty());
    }

    #[test]
    fn test_gguf_transformer_layer_attn_norm() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).unwrap();
        let transformer = GGUFTransformer::from_gguf(&model, &data).unwrap();

        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }
}
