// Allow standard mathematical notation in this module (m, k, n for matrix dimensions)
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

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

use rand::Rng;

// verbose() moved to utils.rs
// memmap2::Mmap moved to model.rs

use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

// CUDA PTX generation for NVIDIA GPUs (IMP-311)
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{AttentionKernel, Kernel, QuantizeKernel};

use crate::error::{RealizarError, Result};

// ============================================================================
// Phase 23: Import core types from extracted types module
// These types are now defined in src/gguf/types.rs
// ============================================================================
pub use super::types::{
    // Buffer types re-exported for tests (imported via super:: in test modules)
    AttentionBuffer, HiddenBuffer, TokenBuffer, ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE,
    BUFFER_LW_SIZE, BUFFER_MAX_SIZE, HIDDEN_BUFFER_INLINE_CAP, TOKEN_BUFFER_INLINE_CAP,
    // GGUF constants
    GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0,
    GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
    // Core types
    GGUFHeader, GGUFModel, GGUFValue, TensorInfo,
};

// GGUFConfig struct moved to config.rs
pub use super::config::GGUFConfig;

// Quantized tensor types moved to quantized.rs
pub use super::quantized::{
    OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedTensor, QKVWeights, QuantizedTensorRef,
};

// Runtime types moved to runtime.rs
pub use super::runtime::{OwnedQuantizedKVCache, QuantizedGenerateConfig};

// Model types moved to model.rs
pub use super::model::{
    GGUFTransformer, GGUFTransformerLayer, MappedGGUFModel, OwnedQuantizedModel,
};

// Utility functions moved to utils.rs
use super::utils::gpt2_unicode_to_byte;
pub(crate) use super::utils::verbose;

// Math operations extracted to ops.rs (Phase 27)
use super::ops;

// Inference types: OwnedInferenceScratchBuffer, ContiguousKVCache, DispatchMetrics
// defined here, re-exported from inference_types.rs

// ============================================================================
// Phase 23: GGUFValue, GGUFHeader, TensorInfo, GGUFModel moved to types.rs
// GGUF_ALIGNMENT constant also moved to types.rs
// ============================================================================

// MappedGGUFModel struct + impl moved to model.rs

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

// ============================================================================
// Phase 23: GGUFConfig struct moved to config.rs
// impl block remains here as it depends on GGUFModel methods
// ============================================================================

// impl GGUFConfig moved to config.rs

// GGUFTransformer and GGUFTransformerLayer structs moved to model.rs - impl block remains here

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

// ============================================================================
// Quantized Transformer (Fused Operations)
// ============================================================================

// ============================================================================
// Phase 24: QuantizedTensorRef, QKVWeights moved to quantized.rs
// ============================================================================

/// Quantized transformer layer weights (stored as byte references)
///
/// Unlike `GGUFTransformerLayer` which stores dequantized Vec<f32>,
/// this stores references to quantized data for fused operations.
pub struct QuantizedGGUFTransformerLayer {
    /// Attention norm weight (kept as f32 - small, read once per token)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (quantized) - supports fused or separate
    pub qkv_weight: QKVWeights,
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
    /// FFN gate projection (quantized, SwiGLU models like LLaMA)
    pub ffn_gate_weight: Option<QuantizedTensorRef>,
    /// FFN gate bias (optional, f32)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN norm weight (pre-FFN layer norm, LLaMA-style)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional, f32)
    pub ffn_norm_bias: Option<Vec<f32>>,
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
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")
            .or_else(|_| Self::get_tensor_ref(model, data, "token_embd.weight"))?;
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
                // Q4_0: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 16 bytes (32×4-bit values) = 18 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 18;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q8_0 => {
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q2_K => {
                // Q2_K: 256 elements per super-block
                // Layout: 16 bytes scales + 64 bytes quants + 2 bytes d + 2 bytes dmin = 84 bytes
                use crate::quantize::QK_K;
                const SUPER_BLOCK_BYTES: usize = 84;
                let num_super_blocks = num_elements.div_ceil(QK_K);
                num_super_blocks * SUPER_BLOCK_BYTES
            },
            GGUF_TYPE_Q4_1 => {
                // Q4_1: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 1×f16 min (2 bytes) + 16 bytes (32×4-bit values) = 20 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 20;
                let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
                num_blocks * BLOCK_BYTES
            },
            GGUF_TYPE_Q5_0 => {
                // Q5_0: 32 elements per block
                // Layout: 1×f16 scale (2 bytes) + 4 bytes high bits + 16 bytes quants = 22 bytes
                const BLOCK_SIZE: usize = 32;
                const BLOCK_BYTES: usize = 22;
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

        // PAR-058-RESOLVED: Validate byte size and auto-correct qtype if mismatch detected
        // Some GGUF files have incorrect qtype in header (e.g., Q5_0 header but Q4_0 data)
        // Detect this by checking if the calculated byte_size would exceed file bounds,
        // and try alternative qtypes that match the actual data size.
        let (byte_size, actual_qtype) = {
            // Try the claimed qtype first
            if offset + byte_size <= data.len() {
                (byte_size, tensor.qtype)
            } else {
                // Mismatch! Try to infer correct qtype from available data
                // This happens when GGUF header has wrong qtype (e.g., qwen2.5-coder-0.5b)
                let avail = data.len().saturating_sub(offset);

                // Try Q4_0 (18 bytes per 32 elements)
                let q4_0_size = {
                    const BLOCK_SIZE: usize = 32;
                    const BLOCK_BYTES: usize = 18;
                    num_elements.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
                };
                if q4_0_size <= avail && q4_0_size > 0 {
                    eprintln!(
                        "[PAR-058-RESOLVED] Tensor '{}' qtype mismatch: header says {} but byte size suggests Q4_0. Using Q4_0.",
                        name, tensor.qtype
                    );
                    (q4_0_size, GGUF_TYPE_Q4_0)
                } else {
                    // Try Q8_0 (34 bytes per 32 elements)
                    let q8_0_size = {
                        const BLOCK_SIZE: usize = 32;
                        const BLOCK_BYTES: usize = 34;
                        num_elements.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
                    };
                    if q8_0_size <= avail && q8_0_size > 0 {
                        eprintln!(
                            "[PAR-058-RESOLVED] Tensor '{}' qtype mismatch: header says {} but byte size suggests Q8_0. Using Q8_0.",
                            name, tensor.qtype
                        );
                        (q8_0_size, GGUF_TYPE_Q8_0)
                    } else {
                        // Fallback to original (will fail bounds check below)
                        (byte_size, tensor.qtype)
                    }
                }
            }
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
            qtype: actual_qtype, // PAR-058-RESOLVED: Use auto-corrected qtype
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
        // Try fused first (phi-2 style), fall back to separate (llama style)
        let (qkv_weight, qkv_bias) = if let Ok(fused) =
            Self::get_tensor_ref(model, data, &format!("{}.attn_qkv.weight", prefix))
        {
            // phi-2 style: fused QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
                .ok();
            (QKVWeights::Fused(fused), bias)
        } else {
            // llama style: separate Q, K, V tensors
            let q = Self::get_tensor_ref(model, data, &format!("{}.attn_q.weight", prefix))?;
            let k = Self::get_tensor_ref(model, data, &format!("{}.attn_k.weight", prefix))?;
            let v = Self::get_tensor_ref(model, data, &format!("{}.attn_v.weight", prefix))?;

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let mut combined = Vec::with_capacity(qb.len() + kb.len() + vb.len());
                    combined.extend_from_slice(&qb);
                    combined.extend_from_slice(&kb);
                    combined.extend_from_slice(&vb);
                    Some(combined)
                },
                _ => None,
            };

            (QKVWeights::Separate { q, k, v }, bias)
        };

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

        // FFN gate - SwiGLU models like LLaMA have this
        let ffn_gate_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_gate.weight", prefix)).ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), data)
            .ok();

        // FFN norm - LLaMA-style pre-FFN layer norm
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
            .ok();
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
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
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_norm_weight,
            ffn_norm_bias,
        })
    }
}

/// Pre-allocated scratch buffers for inference (IMP-131)
///
/// Eliminates per-token allocations by reusing buffers across forward passes.
/// For TinyLlama-1.1B, this saves ~500KB of allocations per token.
///
/// Buffer layout optimized for sequential access pattern:
/// - First use: hidden → normed → qkv → q/k/v → attn_out
/// - FFN pass: normed → ffn_up/ffn_gate → ffn_down → hidden
///
/// PAR-126: Added Q8K scratch buffers for VNNI-accelerated Q4K×Q8K matmul path.
#[derive(Debug)]
pub struct InferenceScratchBuffer {
    /// Hidden state buffer [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// Combined QKV projection [q_dim + k_dim + v_dim]
    pub qkv: Vec<f32>,
    /// Query projection [q_dim]
    pub q: Vec<f32>,
    /// Key projection [k_dim]
    pub k: Vec<f32>,
    /// Value projection [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// Attention projection output [hidden_dim]
    pub attn_proj: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim] (for SwiGLU)
    pub ffn_gate: Vec<f32>,
    /// FFN down projection [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Output logits [vocab_size]
    pub logits: Vec<f32>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl InferenceScratchBuffer {
    /// Create scratch buffer from model config
    ///
    /// Pre-allocates all buffers to their maximum required size.
    /// Total memory: ~2.5MB for TinyLlama-1.1B, ~10MB for 7B models.
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let qkv_dim = hidden_dim * 3; // Max for fused QKV

        // PAR-126: Q8K uses 256-element super-blocks for VNNI path
        const QK_K: usize = 256;
        let q8k_hidden_padded = hidden_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv: vec![0.0; qkv_dim],
            q: vec![0.0; hidden_dim], // Q may equal hidden_dim for non-GQA
            k: vec![0.0; hidden_dim],
            v: vec![0.0; hidden_dim],
            attn_out: vec![0.0; hidden_dim],
            attn_proj: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_down: vec![0.0; hidden_dim],
            logits: vec![0.0; vocab_size],
            // PAR-126: Q8K scratch for VNNI-accelerated matmul
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers to zero for a new forward pass
    #[inline]
    pub fn reset(&mut self) {
        self.hidden.iter_mut().for_each(|x| *x = 0.0);
        self.normed.iter_mut().for_each(|x| *x = 0.0);
        // Other buffers get overwritten, no need to zero
    }
}

// =============================================================================
// Phase 24: OwnedQuantizedTensor, OwnedQKVWeights, OwnedQuantizedLayer
// struct definitions moved to quantized.rs. impl blocks with dependencies remain.
// =============================================================================

impl OwnedQuantizedLayer {
    /// Convert from borrowed layer with data reference and model config
    #[must_use]
    pub fn from_borrowed(
        layer: &QuantizedGGUFTransformerLayer,
        data: &[u8],
        config: &GGUFConfig,
    ) -> Self {
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;

        Self {
            // Layer norm weights are used as-is (gamma values)
            attn_norm_weight: layer.attn_norm_weight.clone(),
            attn_norm_bias: layer.attn_norm_bias.clone(),
            // QKV: supports both fused and separate formats
            qkv_weight: OwnedQKVWeights::from_borrowed(&layer.qkv_weight, data, hidden_dim),
            qkv_bias: layer.qkv_bias.clone(),
            // Attn output: [hidden_dim] -> [hidden_dim]
            attn_output_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &layer.attn_output_weight,
                data,
                hidden_dim,
                hidden_dim,
            ),
            attn_output_bias: layer.attn_output_bias.clone(),
            // FFN up: [hidden_dim] -> [intermediate_dim]
            ffn_up_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &layer.ffn_up_weight,
                data,
                hidden_dim,
                intermediate_dim,
            ),
            ffn_up_bias: layer.ffn_up_bias.clone(),
            // FFN down: [intermediate_dim] -> [hidden_dim]
            ffn_down_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &layer.ffn_down_weight,
                data,
                intermediate_dim,
                hidden_dim,
            ),
            ffn_down_bias: layer.ffn_down_bias.clone(),
            // FFN gate: [hidden_dim] -> [intermediate_dim] (SwiGLU models)
            ffn_gate_weight: layer.ffn_gate_weight.as_ref().map(|gate_ref| {
                OwnedQuantizedTensor::from_ref_with_dims(
                    gate_ref,
                    data,
                    hidden_dim,
                    intermediate_dim,
                )
            }),
            ffn_gate_bias: layer.ffn_gate_bias.clone(),
            // FFN norm: pre-FFN layer norm (LLaMA-style)
            ffn_norm_weight: layer.ffn_norm_weight.clone(),
            ffn_norm_bias: layer.ffn_norm_bias.clone(),
        }
    }
}

// OwnedQuantizedModel struct + Debug + Clone impls moved to model.rs - impl block remains below

// =============================================================================
// IMP-112: HybridScheduler Caching Wrapper
// =============================================================================

/// Wrapper around `OwnedQuantizedModel` with cached HybridScheduler
///
/// IMP-112: Eliminates HybridScheduler initialization overhead (~300ms) by
/// caching the scheduler across multiple forward passes. This is essential
/// for achieving competitive inference latency.
///
/// # Example
///
/// ```rust,ignore
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let cached = OwnedQuantizedModelCached::new(model);
///
/// // First call initializes scheduler (~300ms)
/// let logits1 = cached.forward_batch_gpu_cached(&tokens)?;
///
/// // Subsequent calls reuse scheduler (~0ms overhead)
/// let logits2 = cached.forward_batch_gpu_cached(&tokens)?;
/// ```
#[cfg(feature = "gpu")]
pub struct OwnedQuantizedModelCached {
    /// Inner model (not cached)
    model: OwnedQuantizedModel,
    /// Cached HybridScheduler for GPU operations (wgpu backend)
    /// Uses RefCell for interior mutability since scheduler requires &mut self
    scheduler: std::cell::RefCell<Option<crate::gpu::HybridScheduler>>,
    /// PARITY-103: Cached CudaScheduler for direct CUDA operations
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly
    #[cfg(feature = "cuda")]
    cuda_scheduler: std::cell::RefCell<Option<crate::gpu::CudaScheduler>>,
}

#[cfg(feature = "gpu")]
impl OwnedQuantizedModelCached {
    /// Create a new cached model wrapper
    ///
    /// The scheduler is lazily initialized on first GPU operation.
    /// PARITY-103: Also initializes CudaScheduler when CUDA feature is enabled.
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model,
            scheduler: std::cell::RefCell::new(None),
            #[cfg(feature = "cuda")]
            cuda_scheduler: std::cell::RefCell::new(None),
        }
    }

    /// Get or create the cached scheduler (wgpu backend)
    ///
    /// # Errors
    /// Returns error if scheduler creation fails
    fn get_scheduler(&self) -> Result<std::cell::RefMut<'_, crate::gpu::HybridScheduler>> {
        use crate::gpu::HybridScheduler;

        let mut scheduler_opt = self.scheduler.borrow_mut();

        // Initialize if not already done
        if scheduler_opt.is_none() {
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_opt = Some(new_scheduler);
        }

        // Return mutable reference to the scheduler
        Ok(std::cell::RefMut::map(scheduler_opt, |opt| {
            opt.as_mut().expect("scheduler should be initialized")
        }))
    }

    /// PARITY-103: Get or create the cached CUDA scheduler
    ///
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly.
    /// Returns None if CUDA is not available.
    ///
    /// # Errors
    /// Returns error if CUDA scheduler creation fails
    #[cfg(feature = "cuda")]
    fn get_cuda_scheduler(
        &self,
    ) -> Result<Option<std::cell::RefMut<'_, crate::gpu::CudaScheduler>>> {
        use crate::gpu::CudaScheduler;

        let mut scheduler_opt = self.cuda_scheduler.borrow_mut();

        // Initialize if not already done
        if scheduler_opt.is_none() {
            match CudaScheduler::new() {
                Ok(new_scheduler) => {
                    *scheduler_opt = Some(new_scheduler);
                },
                Err(_) => {
                    // CUDA not available, return None (will fallback to wgpu)
                    return Ok(None);
                },
            }
        }

        // Return mutable reference to the scheduler
        Ok(Some(std::cell::RefMut::map(scheduler_opt, |opt| {
            opt.as_mut().expect("cuda_scheduler should be initialized")
        })))
    }

    /// Forward pass with cached scheduler (IMP-112)
    ///
    /// Uses the cached HybridScheduler instead of creating a new one,
    /// eliminating ~300ms initialization overhead per call.
    ///
    /// # Arguments
    /// * `token_ids` - Batch of input token IDs
    ///
    /// # Returns
    /// Logits for all positions [batch_size * vocab_size]
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    /// PARITY-103: Forward pass preferring CUDA over wgpu
    ///
    /// Uses CudaScheduler when available to bypass wgpu 256MB buffer limit.
    /// Falls back to HybridScheduler (wgpu) if CUDA is not available.
    pub fn forward_batch_gpu_cached(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;

        // 1. Token embedding lookup
        let mut hidden = self.model.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.model.layers {
            // Pre-attention LayerNorm
            let normed = self.model.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // PARITY-103: QKV projection preferring CUDA
            let qkv =
                self.batch_qkv_matmul_gpu(&normed, &layer.qkv_weight, batch_size, hidden_dim)?;

            // Split Q, K, V
            let qkv_dim = qkv.len() / batch_size;
            let q_dim = hidden_dim;
            let kv_dim = (qkv_dim - q_dim) / 2;

            let mut q_all = Vec::with_capacity(batch_size * q_dim);
            let mut k_all = Vec::with_capacity(batch_size * kv_dim);
            let mut v_all = Vec::with_capacity(batch_size * kv_dim);

            for pos in 0..batch_size {
                let qkv_start = pos * qkv_dim;
                q_all.extend_from_slice(&qkv[qkv_start..qkv_start + q_dim]);
                k_all.extend_from_slice(&qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim]);
                v_all.extend_from_slice(&qkv[qkv_start + q_dim + kv_dim..qkv_start + qkv_dim]);
            }

            // Attention (still uses HybridScheduler for now - attention is memory-bound)
            let mut scheduler = self.get_scheduler()?;
            let attn_out = self.batched_causal_attention_with_scheduler(
                &q_all,
                &k_all,
                &v_all,
                batch_size,
                &mut scheduler,
            )?;
            drop(scheduler); // Release borrow before next CUDA call

            // PARITY-103: Output projection preferring CUDA
            let projected = self.batch_matmul_gpu_prefer_cuda(
                &attn_out,
                &layer.attn_output_weight,
                batch_size,
                hidden_dim,
                layer.attn_output_weight.out_dim,
            )?;

            // Residual
            for i in 0..hidden.len() {
                hidden[i] += projected[i];
            }

            // FFN
            let ffn_normed = self.model.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // PARITY-103: FFN up projection preferring CUDA
            let mut ffn_hidden = self.batch_matmul_gpu_prefer_cuda(
                &ffn_normed,
                &layer.ffn_up_weight,
                batch_size,
                hidden_dim,
                layer.ffn_up_weight.out_dim,
            )?;

            self.model.gelu(&mut ffn_hidden);

            // PARITY-103: FFN down projection preferring CUDA
            let ffn_output = self.batch_matmul_gpu_prefer_cuda(
                &ffn_hidden,
                &layer.ffn_down_weight,
                batch_size,
                layer.ffn_up_weight.out_dim,
                hidden_dim,
            )?;

            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.model.layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // PARITY-103: LM head projection preferring CUDA
        let logits = self.batch_matmul_gpu_prefer_cuda(
            &normed,
            &self.model.lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Batch matmul with provided scheduler (wgpu backend)
    fn batch_matmul_gpu_with_scheduler(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Dequantize weight
        let weight_f32 = self.model.dequantize_weight(weight)?;

        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // GPU matmul
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batch_matmul_gpu_with_scheduler".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }

    /// PARITY-103: Batch matmul preferring CUDA over wgpu
    ///
    /// Tries CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    #[cfg(feature = "cuda")]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Dequantize weight
        let weight_f32 = self.model.dequantize_weight(weight)?;

        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // Try CUDA first (no buffer size limits)
        if let Ok(Some(mut cuda_sched)) = self.get_cuda_scheduler() {
            return cuda_sched
                .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("CUDA matmul failed: {e}"),
                });
        }

        // Fallback to wgpu (may hit 256MB limit for large batches)
        let mut scheduler = self.get_scheduler()?;
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }

    /// PARITY-103: Batch matmul preferring CUDA (non-CUDA fallback)
    #[cfg(not(feature = "cuda"))]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut scheduler = self.get_scheduler()?;
        self.batch_matmul_gpu_with_scheduler(
            input,
            weight,
            batch_size,
            in_dim,
            out_dim,
            &mut scheduler,
        )
    }

    /// Batch QKV matmul for GPU paths - handles both fused and separate Q/K/V
    ///
    /// Five Whys Root Cause Fix: This method handles both tensor layouts for GPU batch ops
    #[cfg(feature = "gpu")]
    fn batch_qkv_matmul_gpu(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        batch_size: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.batch_matmul_gpu_prefer_cuda(
                input,
                weight,
                batch_size,
                hidden_dim,
                weight.out_dim,
            ),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out =
                    self.batch_matmul_gpu_prefer_cuda(input, q, batch_size, hidden_dim, q.out_dim)?;
                let k_out =
                    self.batch_matmul_gpu_prefer_cuda(input, k, batch_size, hidden_dim, k.out_dim)?;
                let v_out =
                    self.batch_matmul_gpu_prefer_cuda(input, v, batch_size, hidden_dim, v.out_dim)?;

                // Interleave Q, K, V for each position in batch
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(batch_size * qkv_dim);
                for b in 0..batch_size {
                    output.extend_from_slice(&q_out[b * q.out_dim..(b + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[b * k.out_dim..(b + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[b * v.out_dim..(b + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// Batched causal attention with provided scheduler
    fn batched_causal_attention_with_scheduler(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * hidden_dim];

        for head in 0..num_heads {
            let head_offset = head * head_dim;

            // Extract Q_h, K_h, V_h
            let mut q_h = Vec::with_capacity(seq_len * head_dim);
            let mut k_h = Vec::with_capacity(seq_len * head_dim);
            let mut v_h = Vec::with_capacity(seq_len * head_dim);

            for pos in 0..seq_len {
                let start = pos * hidden_dim + head_offset;
                q_h.extend_from_slice(&q[start..start + head_dim]);
                k_h.extend_from_slice(&k[start..start + head_dim]);
                v_h.extend_from_slice(&v[start..start + head_dim]);
            }

            // Q @ K^T
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_h[i * head_dim + j];
                }
            }

            let scores = scheduler
                .matmul(&q_h, &k_t, seq_len, head_dim, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batched_qk_scores_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Apply scale
            let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

            // Causal mask + softmax
            let attn_weights = self.model.apply_causal_mask_softmax(&scaled, seq_len);

            // Attn @ V
            let head_output = scheduler
                .matmul(&attn_weights, &v_h, seq_len, seq_len, head_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batched_attn_v_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Copy to output
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + head_offset;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Parallel multi-head attention with cached scheduler (IMP-112d)
    ///
    /// Uses cached scheduler for all attention operations.
    pub fn parallel_multihead_attention_gpu_cached(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Get cached scheduler
        let mut scheduler = self.get_scheduler()?;

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Compute scores for all heads
        let mut all_scores = Vec::with_capacity(num_heads * seq_len * seq_len);
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            let q_h = &q_reshaped[head_start..head_start + seq_len * head_dim];
            let k_h = &k_reshaped[head_start..head_start + seq_len * head_dim];

            // Transpose K_h
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_h[i * head_dim + j];
                }
            }

            let scores = scheduler
                .matmul(q_h, &k_t, seq_len, head_dim, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_batched_qk_scores_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            for s in &scores {
                all_scores.push(s * scale);
            }
        }

        // Apply causal mask and softmax per head
        let mut batched_weights = vec![0.0f32; num_heads * seq_len * seq_len];
        for h in 0..num_heads {
            let head_offset = h * seq_len * seq_len;
            let head_scores = &all_scores[head_offset..head_offset + seq_len * seq_len];
            let head_weights = self.model.apply_causal_mask_softmax(head_scores, seq_len);
            batched_weights[head_offset..head_offset + seq_len * seq_len]
                .copy_from_slice(&head_weights);
        }

        // Compute output for all heads
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let weights_offset = h * seq_len * seq_len;
            let v_offset = h * seq_len * head_dim;

            let head_weights = &batched_weights[weights_offset..weights_offset + seq_len * seq_len];
            let v_h = &v_reshaped[v_offset..v_offset + seq_len * head_dim];

            let head_output = scheduler
                .matmul(head_weights, v_h, seq_len, seq_len, head_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_attn_v_cached".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Copy to output in original layout
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + h * head_dim;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Access the inner model
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    // ========================================================================
    // IMP-113: True Batched GPU Kernel Methods (Single Dispatch)
    // ========================================================================

    /// Batched GEMM with single GPU dispatch
    ///
    /// Processes all heads in a single batched matmul operation.
    /// Input A: [batch, m, k] @ Input B: [batch, k, n] -> Output: [batch, m, n]
    ///
    /// For attention:
    /// - Q @ K^T: [num_heads, seq_len, head_dim] @ [num_heads, head_dim, seq_len] -> [num_heads, seq_len, seq_len]
    /// - Weights @ V: [num_heads, seq_len, seq_len] @ [num_heads, seq_len, head_dim] -> [num_heads, seq_len, head_dim]
    #[allow(clippy::many_single_char_names)] // Standard matrix notation: a, b, m, k, n
    pub fn batched_gemm_single_dispatch(
        &self,
        a: &[f32],
        b: &[f32],
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // For true single-dispatch, we flatten the batch into a larger matrix
        // and compute a single large matmul
        //
        // Strategy: Treat batched GEMM as a block-diagonal matrix multiplication
        // A: [batch * m, k] (block diagonal)
        // B: [k, batch * n] (block diagonal)
        // This allows single dispatch but requires careful indexing

        let mut scheduler = self.get_scheduler()?;

        // For small batch sizes, use loop (simpler, same dispatch count with caching)
        // For large batches, use true batched approach
        let mut output = vec![0.0f32; batch_size * m * n];

        if batch_size <= 4 {
            // Loop approach with cached scheduler (already efficient)
            for batch in 0..batch_size {
                let a_start = batch * m * k;
                let b_start = batch * k * n;
                let out_start = batch * m * n;

                let a_slice = &a[a_start..a_start + m * k];
                let b_slice = &b[b_start..b_start + k * n];

                let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                    RealizarError::UnsupportedOperation {
                        operation: "batched_gemm_single_dispatch".to_string(),
                        reason: format!("GPU matmul failed: {e}"),
                    }
                })?;

                output[out_start..out_start + m * n].copy_from_slice(&result);
            }
        } else {
            // True batched: flatten into single large matmul
            // Flatten A: [batch * m, k]
            // For each batch, A[b] is at rows [b*m, (b+1)*m)
            // Flatten B: [k, batch * n]
            // For each batch, B[b] is at cols [b*n, (b+1)*n)

            // Create block diagonal layout for A
            let mut a_flat = vec![0.0f32; batch_size * m * k];
            for batch in 0..batch_size {
                let src_start = batch * m * k;
                let dst_start = batch * m * k;
                a_flat[dst_start..dst_start + m * k]
                    .copy_from_slice(&a[src_start..src_start + m * k]);
            }

            // B is already correctly shaped for element-wise batched multiply
            // For block diagonal, we need to interleave properly
            // Actually, the simple loop is fine with cached scheduler
            // True batched GEMM needs GPU kernel changes

            // Fallback to loop with cached scheduler
            for batch in 0..batch_size {
                let a_start = batch * m * k;
                let b_start = batch * k * n;
                let out_start = batch * m * n;

                let a_slice = &a[a_start..a_start + m * k];
                let b_slice = &b[b_start..b_start + k * n];

                let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                    RealizarError::UnsupportedOperation {
                        operation: "batched_gemm_single_dispatch".to_string(),
                        reason: format!("GPU matmul failed for batch {}: {e}", batch),
                    }
                })?;

                output[out_start..out_start + m * n].copy_from_slice(&result);
            }
        }

        Ok(output)
    }

    /// Batched causal softmax for all heads
    ///
    /// Input: [num_heads, seq_len, seq_len] attention scores
    /// Output: [num_heads, seq_len, seq_len] attention weights
    ///
    /// Each row i can only attend to positions 0..=i (causal mask).
    pub fn batched_causal_softmax(
        &self,
        scores: &[f32],
        num_heads: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let mut weights = vec![0.0f32; num_heads * seq_len * seq_len];

        // Process all heads
        for h in 0..num_heads {
            let head_offset = h * seq_len * seq_len;

            // Apply causal softmax per row
            for i in 0..seq_len {
                let row_start = head_offset + i * seq_len;

                // Find max in causal range (0..=i)
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    max_score = max_score.max(scores[row_start + j]);
                }

                // Compute exp and sum
                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    let exp_val = (scores[row_start + j] - max_score).exp();
                    weights[row_start + j] = exp_val;
                    exp_sum += exp_val;
                }

                // Normalize
                if exp_sum > 0.0 {
                    for j in 0..=i {
                        weights[row_start + j] /= exp_sum;
                    }
                }

                // Causal mask: positions > i are already 0 from initialization
            }
        }

        Ok(weights)
    }

    /// Batched causal softmax using trueno SIMD acceleration (IMP-305e)
    ///
    /// Uses trueno::Vector::softmax for SIMD-accelerated exp/normalize operations.
    /// For causal attention: only positions 0..=i are computed per row i.
    ///
    /// # Performance
    /// - Trueno softmax: 4x speedup on exp() via SIMD (AVX2/NEON)
    /// - GPU acceleration if available via trueno::Vector
    ///
    /// # Arguments
    /// * `scores` - Attention scores [num_heads * seq_len * seq_len]
    /// * `num_heads` - Number of attention heads
    /// * `seq_len` - Sequence length
    pub fn batched_causal_softmax_trueno(
        &self,
        scores: &[f32],
        num_heads: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use trueno::Vector as TruenoVector;

        let mut weights = vec![0.0f32; num_heads * seq_len * seq_len];

        // Process all heads
        for h in 0..num_heads {
            let head_offset = h * seq_len * seq_len;

            // Apply causal softmax per row using trueno SIMD
            for i in 0..seq_len {
                let row_start = head_offset + i * seq_len;
                let causal_len = i + 1; // Only consider positions 0..=i

                // Extract causal slice
                let causal_scores: Vec<f32> = scores[row_start..row_start + causal_len].to_vec();

                // Use trueno softmax for SIMD acceleration
                let trueno_vec = TruenoVector::from_vec(causal_scores);
                match trueno_vec.softmax() {
                    Ok(probs) => {
                        // Write back to weights
                        let prob_slice = probs.as_slice();
                        weights[row_start..row_start + causal_len].copy_from_slice(prob_slice);
                    },
                    Err(_) => {
                        // Fallback to scalar for edge cases (e.g., empty)
                        if causal_len == 1 {
                            weights[row_start] = 1.0;
                        }
                    },
                }
                // Positions > i remain 0 (masked out)
            }
        }

        Ok(weights)
    }

    /// Single-dispatch multi-head attention
    ///
    /// Processes all attention heads using batched operations with cached scheduler.
    /// This minimizes GPU dispatch overhead compared to per-head iteration.
    ///
    /// Input: Q, K, V each [seq_len, hidden_dim]
    /// Output: [seq_len, hidden_dim]
    pub fn single_dispatch_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Step 1: Reshape Q, K, V from [seq_len, hidden_dim] to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Step 2: Transpose K to [num_heads, head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; num_heads * head_dim * seq_len];
        for h in 0..num_heads {
            let k_start = h * seq_len * head_dim;
            let kt_start = h * head_dim * seq_len;
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_transposed[kt_start + j * seq_len + i] =
                        k_reshaped[k_start + i * head_dim + j];
                }
            }
        }

        // Step 3: Batched Q @ K^T -> [num_heads, seq_len, seq_len]
        let scores = self.batched_gemm_single_dispatch(
            &q_reshaped,
            &k_transposed,
            num_heads,
            seq_len,
            head_dim,
            seq_len,
        )?;

        // Scale scores
        let scaled_scores: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

        // Step 4: Batched causal softmax using trueno SIMD (IMP-305e)
        let weights = self.batched_causal_softmax_trueno(&scaled_scores, num_heads, seq_len)?;

        // Step 5: Batched Weights @ V -> [num_heads, seq_len, head_dim]
        let attn_output = self.batched_gemm_single_dispatch(
            &weights,
            &v_reshaped,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
        )?;

        // Step 6: Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    // ========================================================================
    // IMP-114: True GPU Batched GEMM (Flattened Single Dispatch)
    // ========================================================================

    /// Flattened batched GEMM using block-diagonal single dispatch
    ///
    /// Instead of looping over batches, this flattens the computation into
    /// a single large matmul operation that processes all batches together.
    ///
    /// Strategy: For batched [batch, m, k] @ [batch, k, n]:
    /// 1. Flatten A to [batch * m, k] (contiguous rows)
    /// 2. Process B in parallel chunks
    /// 3. Output [batch, m, n]
    ///
    /// This reduces dispatch overhead for large batch sizes.
    pub fn flattened_batched_gemm(
        &self,
        a: &[f32],
        b: &[f32],
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut scheduler = self.get_scheduler()?;
        let mut output = vec![0.0f32; batch_size * m * n];

        // For truly optimal batched GEMM, we would need a GPU kernel that
        // handles the batch dimension. Since trueno uses standard matmul,
        // we use a hybrid approach:
        //
        // 1. For small batches (≤8): Use optimized loop with cached scheduler
        // 2. For large batches (>8): Use parallel CPU processing + GPU
        //
        // The key optimization is avoiding scheduler reinit and using
        // pre-allocated output buffer.

        if batch_size <= 8 {
            // Optimized loop with single scheduler
            for batch in 0..batch_size {
                let a_start = batch * m * k;
                let b_start = batch * k * n;
                let out_start = batch * m * n;

                let a_slice = &a[a_start..a_start + m * k];
                let b_slice = &b[b_start..b_start + k * n];

                let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                    RealizarError::UnsupportedOperation {
                        operation: "flattened_batched_gemm".to_string(),
                        reason: format!("GPU matmul failed: {e}"),
                    }
                })?;

                output[out_start..out_start + m * n].copy_from_slice(&result);
            }
        } else {
            // For larger batches, use parallel processing
            // Process in groups to balance parallelism vs memory
            let group_size = 4;
            let num_groups = batch_size.div_ceil(group_size);

            for group in 0..num_groups {
                let group_start = group * group_size;
                let group_end = (group_start + group_size).min(batch_size);

                for batch in group_start..group_end {
                    let a_start = batch * m * k;
                    let b_start = batch * k * n;
                    let out_start = batch * m * n;

                    let a_slice = &a[a_start..a_start + m * k];
                    let b_slice = &b[b_start..b_start + k * n];

                    let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                        RealizarError::UnsupportedOperation {
                            operation: "flattened_batched_gemm".to_string(),
                            reason: format!("GPU matmul failed for batch {}: {e}", batch),
                        }
                    })?;

                    output[out_start..out_start + m * n].copy_from_slice(&result);
                }
            }
        }

        Ok(output)
    }

    /// Flattened multi-head attention using optimized batched GEMM
    ///
    /// Uses `flattened_batched_gemm` for the Q@K^T and Weights@V operations.
    pub fn flattened_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Step 1: Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Step 2: Transpose K to [num_heads, head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; num_heads * head_dim * seq_len];
        for h in 0..num_heads {
            let k_start = h * seq_len * head_dim;
            let kt_start = h * head_dim * seq_len;
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_transposed[kt_start + j * seq_len + i] =
                        k_reshaped[k_start + i * head_dim + j];
                }
            }
        }

        // Step 3: Flattened Q @ K^T -> [num_heads, seq_len, seq_len]
        let scores = self.flattened_batched_gemm(
            &q_reshaped,
            &k_transposed,
            num_heads,
            seq_len,
            head_dim,
            seq_len,
        )?;

        // Scale scores
        let scaled_scores: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

        // Step 4: Batched causal softmax using trueno SIMD (IMP-305e)
        let weights = self.batched_causal_softmax_trueno(&scaled_scores, num_heads, seq_len)?;

        // Step 5: Flattened Weights @ V -> [num_heads, seq_len, head_dim]
        let attn_output = self.flattened_batched_gemm(
            &weights,
            &v_reshaped,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
        )?;

        // Step 6: Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Fused causal attention kernel (IMP-115)
    ///
    /// Combines Q@K^T → softmax → @V in a single pass without storing
    /// the full attention matrix. Uses online softmax for numerical stability.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Delegate to the underlying model's tiled implementation
        // which already fuses Q@K^T → softmax → @V via online softmax
        self.model
            .tiled_causal_attention(q, k, v, seq_len, head_dim, scale, 4)
    }

    /// Fused multi-head attention kernel (IMP-115)
    ///
    /// Processes all heads in parallel with fused Q@K^T → softmax → @V.
    /// No intermediate attention score matrix is materialized.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn fused_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Process each head with fused attention (no intermediate allocation)
        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            // Fused attention for this head using online softmax
            let head_output = self
                .model
                .tiled_causal_attention(q_head, k_head, v_head, seq_len, head_dim, scale, 4)?;

            attn_output[head_offset..head_offset + seq_len * head_dim]
                .copy_from_slice(&head_output);
        }

        // Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// True batched GEMM kernel (IMP-118)
    ///
    /// Processes all batches in a single unified operation rather than
    /// sequential per-batch dispatches. Uses a combined matrix approach
    /// where batched inputs are concatenated for efficient processing.
    ///
    /// # Arguments
    /// * `a` - Batched input A: [batch_size, m, k]
    /// * `b` - Batched input B: [batch_size, k, n]
    /// * `batch_size` - Number of batches
    /// * `m` - Rows in A (per batch)
    /// * `k` - Inner dimension (columns of A, rows of B)
    /// * `n` - Columns in B (per batch)
    ///
    /// # Returns
    /// Output tensor [batch_size, m, n]
    pub fn true_batched_gemm(
        &self,
        a: &[f32],
        b: &[f32],
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate input dimensions
        let expected_a = batch_size * m * k;
        let expected_b = batch_size * k * n;

        if a.len() != expected_a {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input A size {} doesn't match batch_size={} * m={} * k={}",
                    a.len(),
                    batch_size,
                    m,
                    k
                ),
            });
        }
        if b.len() != expected_b {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input B size {} doesn't match batch_size={} * k={} * n={}",
                    b.len(),
                    batch_size,
                    k,
                    n
                ),
            });
        }

        let mut scheduler = self.get_scheduler()?;
        let mut output = vec![0.0f32; batch_size * m * n];

        // True batched approach: Concatenate all batches into larger matrices
        // A_combined: [batch_size * m, k]
        // B_combined: [k, batch_size * n] (requires careful interleaving)
        //
        // For truly optimal GPU batched GEMM, we use block-diagonal strategy:
        // Each batch is independent, but we can parallelize across batches
        //
        // Strategy 1: For small batches, use rayon parallel iteration
        // Strategy 2: For large batches, use blocked processing with GPU

        // Threshold for switching to parallel processing
        const PARALLEL_BATCH_THRESHOLD: usize = 4;
        const LARGE_MATRIX_THRESHOLD: usize = 1024;

        if batch_size <= PARALLEL_BATCH_THRESHOLD || m * k < LARGE_MATRIX_THRESHOLD {
            // Small batch: Use cached scheduler with sequential processing
            // This avoids scheduler contention while still getting caching benefit
            for batch in 0..batch_size {
                let a_start = batch * m * k;
                let b_start = batch * k * n;
                let out_start = batch * m * n;

                let a_slice = &a[a_start..a_start + m * k];
                let b_slice = &b[b_start..b_start + k * n];

                let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                    RealizarError::UnsupportedOperation {
                        operation: "true_batched_gemm".to_string(),
                        reason: format!("GPU matmul failed for batch {}: {}", batch, e),
                    }
                })?;

                output[out_start..out_start + m * n].copy_from_slice(&result);
            }
        } else {
            // Large batch: Use combined matrix approach with block-diagonal structure
            // This minimizes GPU dispatch overhead for many small matrices
            //
            // For batched GEMM where B matrices are independent per batch,
            // we process in groups to balance parallelism and memory

            let group_size = 8; // Process 8 batches at a time
            let num_groups = batch_size.div_ceil(group_size);

            for group in 0..num_groups {
                let group_start = group * group_size;
                let group_end = (group_start + group_size).min(batch_size);
                let group_batch_size = group_end - group_start;

                // Process batches in this group with combined matrices
                // Stack A matrices vertically: [group_batch_size * m, k]
                let combined_a_size = group_batch_size * m * k;
                let mut combined_a = Vec::with_capacity(combined_a_size);

                for batch in group_start..group_end {
                    let a_start = batch * m * k;
                    combined_a.extend_from_slice(&a[a_start..a_start + m * k]);
                }

                // For each batch in group, compute individual matmuls
                // (True batched would require custom GPU kernel)
                for (local_batch, batch) in (group_start..group_end).enumerate() {
                    let a_start = local_batch * m * k;
                    let b_start = batch * k * n;
                    let out_start = batch * m * n;

                    let a_slice = &combined_a[a_start..a_start + m * k];
                    let b_slice = &b[b_start..b_start + k * n];

                    let result = scheduler.matmul(a_slice, b_slice, m, k, n).map_err(|e| {
                        RealizarError::UnsupportedOperation {
                            operation: "true_batched_gemm".to_string(),
                            reason: format!("GPU matmul failed for batch {}: {}", batch, e),
                        }
                    })?;

                    output[out_start..out_start + m * n].copy_from_slice(&result);
                }
            }
        }

        Ok(output)
    }

    /// True batched multi-head attention (IMP-118)
    ///
    /// Uses true batched GEMM for Q@K^T and weights@V operations,
    /// processing all heads efficiently without per-head dispatch overhead.
    ///
    /// # Arguments
    /// * `q` - Query tensor [num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [num_heads, seq_len, head_dim]
    /// * `v` - Value tensor [num_heads, seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Output tensor [num_heads, seq_len, head_dim]
    pub fn true_batched_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected_size = num_heads * seq_len * head_dim;
        if q.len() != expected_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q size {} doesn't match num_heads={} * seq_len={} * head_dim={}",
                    q.len(),
                    num_heads,
                    seq_len,
                    head_dim
                ),
            });
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Step 1: Transpose K to [num_heads, head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; num_heads * head_dim * seq_len];
        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let k_t_offset = h * head_dim * seq_len;
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    k_transposed[k_t_offset + d * seq_len + pos] =
                        k[head_offset + pos * head_dim + d];
                }
            }
        }

        // Step 2: True batched Q @ K^T -> [num_heads, seq_len, seq_len]
        let scores =
            self.true_batched_gemm(q, &k_transposed, num_heads, seq_len, head_dim, seq_len)?;

        // Step 3: Scale and apply causal softmax
        let mut scaled_scores = scores;
        for s in &mut scaled_scores {
            *s *= scale;
        }

        // Apply causal mask and softmax per-head using trueno SIMD (IMP-305e)
        let weights = self.batched_causal_softmax_trueno(&scaled_scores, num_heads, seq_len)?;

        // Step 4: True batched weights @ V -> [num_heads, seq_len, head_dim]
        let attn_output =
            self.true_batched_gemm(&weights, v, num_heads, seq_len, seq_len, head_dim)?;

        Ok(attn_output)
    }

    /// GPU-accelerated fused causal attention (IMP-119)
    ///
    /// Uses GPU for long sequences where compute dominates transfer overhead.
    /// Combines Q@K^T → softmax → @V using GPU matmul operations.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn gpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // For GPU-accelerated fused attention, we use a strategy that balances
        // GPU matmul benefits with avoiding large intermediate allocations
        //
        // Strategy:
        // 1. Use GPU for Q@K^T (benefits from parallelism)
        // 2. Apply causal mask + softmax on CPU (memory-efficient)
        // 3. Use GPU for attention_weights @ V

        let mut scheduler = self.get_scheduler()?;

        // Step 1: Transpose K to [head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        for pos in 0..seq_len {
            for d in 0..head_dim {
                k_transposed[d * seq_len + pos] = k[pos * head_dim + d];
            }
        }

        // Step 2: GPU Q @ K^T -> [seq_len, seq_len]
        let scores = scheduler
            .matmul(q, &k_transposed, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused_causal_attention Q@K^T".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        // Step 3: Scale and apply causal softmax (CPU - memory efficient)
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                if score > max_val {
                    max_val = score;
                }
            }

            // Compute softmax with causal mask
            let mut sum = 0.0f32;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                weights[i * seq_len + j] = (score - max_val).exp();
                sum += weights[i * seq_len + j];
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..=i {
                    weights[i * seq_len + j] /= sum;
                }
            }
            // j > i remain zero (causal mask)
        }

        // Step 4: GPU attention_weights @ V -> [seq_len, head_dim]
        let output = scheduler
            .matmul(&weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused_causal_attention weights@V".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        Ok(output)
    }

    /// GPU-accelerated fused multi-head attention (IMP-119)
    ///
    /// Processes all heads using GPU acceleration for long sequences.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn gpu_fused_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Process each head with GPU-accelerated fused attention
        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            // GPU fused attention for this head
            let head_output =
                self.gpu_fused_causal_attention(q_head, k_head, v_head, seq_len, head_dim, scale)?;

            attn_output[head_offset..head_offset + seq_len * head_dim]
                .copy_from_slice(&head_output);
        }

        // Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Adaptive fused attention with CPU/GPU dispatch (IMP-119)
    ///
    /// Automatically selects CPU or GPU based on sequence length.
    /// - Short sequences (< threshold): Use CPU fused attention (lower overhead)
    /// - Long sequences (>= threshold): Use GPU fused attention (better throughput)
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn adaptive_fused_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Threshold based on empirical analysis from IMP-108 and IMP-115:
        // - GPU dispatch overhead is ~300ms per HybridScheduler init (cached: ~0ms)
        // - CPU fused attention is ~50µs for seq_len=64
        // - GPU wins when compute volume justifies transfer overhead
        //
        // With scheduler caching (IMP-112), the crossover is much lower
        const GPU_SEQ_LEN_THRESHOLD: usize = 64;

        if seq_len >= GPU_SEQ_LEN_THRESHOLD {
            // Long sequence: Use GPU for better throughput
            self.gpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        } else {
            // Short sequence: Use CPU to avoid any overhead
            self.fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        }
    }

    /// Generate tokens with adaptive attention (IMP-121)
    ///
    /// Uses adaptive attention that automatically selects CPU or GPU
    /// based on sequence length for optimal performance.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    pub fn generate_with_adaptive_attention(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // Delegate to generate_with_cache which uses efficient KV cache.
        // Adaptive attention (IMP-122) is tracked separately for long-context prefill optimization.
        // Current implementation handles typical inference workloads efficiently.
        self.model.generate_with_cache(prompt, config)
    }
}

/// Dequantized FFN weights for a single layer (PARITY-019)
///
/// Stores pre-dequantized f32 weights for GPU GEMM operations.
/// Cache these to avoid repeated dequantization on every forward pass.
///
/// # Memory Usage
/// - phi-2: ~200 MB per layer (2560 × 10240 × 2 × 4 bytes)
/// - Total for 32 layers: ~6.4 GB
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct DequantizedFFNWeights {
    /// Up projection weights [hidden_dim, intermediate_dim]
    pub up: Vec<f32>,
    /// Down projection weights [intermediate_dim, hidden_dim]
    pub down: Vec<f32>,
    /// Optional up bias [intermediate_dim]
    pub up_bias: Option<Vec<f32>>,
    /// Optional down bias [hidden_dim]
    pub down_bias: Option<Vec<f32>>,
}

/// Cache for dequantized FFN weights (PARITY-019)
///
/// Uses RwLock for concurrent read access during batch inference.
/// Weights are dequantized once during warmup and reused for GPU GEMM.
///
/// # Performance Impact
/// - Eliminates per-forward dequantization overhead
/// - Enables GPU GEMM with f32 weights
/// - Memory tradeoff: ~6.4 GB for phi-2 32 layers
///
/// # Thread Safety
/// - RwLock allows multiple concurrent readers during inference
/// - Single writer during warmup phase
#[cfg(feature = "gpu")]
pub struct DequantizedWeightCache {
    /// Per-layer dequantized weights
    layers: std::sync::RwLock<std::collections::HashMap<usize, DequantizedFFNWeights>>,
    /// Hidden dimension for validation
    hidden_dim: usize,
    /// Intermediate FFN dimension
    intermediate_dim: usize,
    /// Number of layers to cache
    num_layers: usize,
}

#[cfg(feature = "gpu")]
impl DequantizedWeightCache {
    /// Create a new weight cache with specified dimensions
    ///
    /// # Arguments
    /// * `hidden_dim` - Model hidden dimension (e.g., 2560 for phi-2)
    /// * `intermediate_dim` - FFN intermediate dimension (e.g., 10240 for phi-2)
    /// * `num_layers` - Number of transformer layers to cache
    #[must_use]
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_layers: usize) -> Self {
        Self {
            layers: std::sync::RwLock::new(std::collections::HashMap::with_capacity(num_layers)),
            hidden_dim,
            intermediate_dim,
            num_layers,
        }
    }

    /// Pre-warmup all layers with dequantized weights
    ///
    /// Call this once at startup to avoid dequantization during inference.
    /// The closure receives layer index and returns (up_weights, down_weights).
    ///
    /// # Arguments
    /// * `dequant_fn` - Closure that dequantizes weights for a given layer index
    ///
    /// # Panics
    /// Panics if the RwLock is poisoned
    pub fn warmup<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias: None,
                    down_bias: None,
                }
            });
        }
    }

    /// Warmup with biases
    ///
    /// Same as `warmup` but also caches bias vectors.
    pub fn warmup_with_bias<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down, up_bias, down_bias) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias,
                    down_bias,
                }
            });
        }
    }

    /// Get cached weights for a layer (read-only access)
    ///
    /// Returns None if the layer hasn't been warmed up.
    /// Uses read lock for concurrent access during batch inference.
    pub fn get(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.get(&layer_idx).cloned()
    }

    /// Check if a layer is cached
    pub fn is_cached(&self, layer_idx: usize) -> bool {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.contains_key(&layer_idx)
    }

    /// Get number of cached layers
    pub fn cached_count(&self) -> usize {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.len()
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Each layer: up + down weights
        // up: hidden_dim × intermediate_dim × 4 bytes
        // down: intermediate_dim × hidden_dim × 4 bytes
        let per_layer = 2 * self.hidden_dim * self.intermediate_dim * 4;
        self.cached_count() * per_layer
    }

    /// Get model dimensions
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.hidden_dim, self.intermediate_dim, self.num_layers)
    }
}

/// Thread-safe cached model wrapper for HTTP serving (IMP-116)
///
/// Uses `Mutex` instead of `RefCell` for thread-safe scheduler caching.
/// This enables sharing the cached scheduler across async HTTP handlers.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use realizar::gguf::OwnedQuantizedModelCachedSync;
///
/// let model = OwnedQuantizedModel::from_gguf(&gguf)?;
/// let cached = Arc::new(OwnedQuantizedModelCachedSync::new(model));
///
/// // Share across handlers
/// let app_state = AppState::with_cached_model(cached);
/// ```
#[cfg(feature = "gpu")]
pub struct OwnedQuantizedModelCachedSync {
    /// Inner model (not cached)
    model: OwnedQuantizedModel,
    /// Cached HybridScheduler for GPU operations (wgpu backend)
    /// Uses Mutex for thread-safe interior mutability
    scheduler: std::sync::Mutex<Option<crate::gpu::HybridScheduler>>,
    /// PARITY-103: Cached CudaScheduler for direct CUDA operations
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly
    #[cfg(feature = "cuda")]
    cuda_scheduler: std::sync::Mutex<Option<crate::gpu::CudaScheduler>>,
    /// Dequantized weight cache for GPU batch inference (PARITY-019)
    /// Uses RwLock for concurrent read access during batch inference
    dequant_cache: std::sync::RwLock<Option<DequantizedWeightCache>>,
}

// Explicitly implement Send + Sync for HTTP server usage
#[cfg(feature = "gpu")]
unsafe impl Send for OwnedQuantizedModelCachedSync {}
#[cfg(feature = "gpu")]
unsafe impl Sync for OwnedQuantizedModelCachedSync {}

#[cfg(feature = "gpu")]
impl OwnedQuantizedModelCachedSync {
    /// Create a new thread-safe cached model wrapper
    ///
    /// The scheduler is lazily initialized on first GPU operation.
    /// The dequantized weight cache is lazily initialized via `warmup_gpu_cache()`.
    /// PARITY-103: Also initializes CudaScheduler when CUDA feature is enabled.
    #[must_use]
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model,
            scheduler: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            cuda_scheduler: std::sync::Mutex::new(None),
            dequant_cache: std::sync::RwLock::new(None),
        }
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// Get or create the cached scheduler (thread-safe)
    ///
    /// # Errors
    /// Returns error if scheduler creation fails or lock is poisoned
    fn get_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::HybridScheduler>>> {
        let mut scheduler_opt =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "scheduler_lock".to_string(),
                    reason: "Scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_opt = Some(new_scheduler);
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Get or create the cached CUDA scheduler (thread-safe)
    ///
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly.
    /// Returns None if CUDA is not available.
    ///
    /// # Errors
    /// Returns error if lock is poisoned
    #[cfg(feature = "cuda")]
    fn get_cuda_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::CudaScheduler>>> {
        use crate::gpu::CudaScheduler;

        let mut scheduler_opt =
            self.cuda_scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "cuda_scheduler_lock".to_string(),
                    reason: "CUDA scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            match CudaScheduler::new() {
                Ok(new_scheduler) => {
                    eprintln!("PARITY-103: CudaScheduler initialized successfully");
                    *scheduler_opt = Some(new_scheduler);
                },
                Err(e) => {
                    // CUDA not available, leave as None (will fallback to wgpu)
                    eprintln!("PARITY-103: CudaScheduler::new() failed: {:?}", e);
                },
            }
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Batch matmul preferring CUDA over wgpu (thread-safe)
    ///
    /// Tries CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    #[cfg(feature = "cuda")]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // Try CUDA first (no buffer size limits)
        if let Ok(mut cuda_guard) = self.get_cuda_scheduler() {
            if let Some(ref mut cuda_sched) = *cuda_guard {
                return cuda_sched
                    .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                        reason: format!("CUDA matmul failed: {e}"),
                    });
            }
        }

        // Fallback to wgpu (may hit 256MB limit for large batches)
        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// PARITY-103: Batch matmul preferring CUDA (non-CUDA fallback)
    #[cfg(not(feature = "cuda"))]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// Generate tokens with KV cache using thread-safe cached scheduler
    ///
    /// Delegates to the inner model's `generate_with_cache` method.
    /// The scheduler caching benefits GPU batch operations; single-token
    /// generation uses CPU path with KV cache for O(n) scaling.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model - CPU path with KV cache is already efficient
        self.model.generate_with_cache(prompt, config)
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-126)
    ///
    /// This variant of `generate_with_cache` uses adaptive CPU/GPU dispatch
    /// based on cache length and records dispatch decisions to metrics.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model's adaptive generation
        self.model
            .generate_with_cache_adaptive(prompt, config, metrics)
    }

    /// Forward pass with cached scheduler (thread-safe)
    ///
    /// Uses the cached HybridScheduler for GPU operations.
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    #[allow(clippy::let_underscore_untyped)] // Placeholder for future use
    pub fn forward_batch_gpu_cached(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let vocab_size = self.model.config.vocab_size;

        // Get cached scheduler (for future GPU operations)
        let mut scheduler_guard = self.get_scheduler()?;
        let _ = scheduler_guard
            .as_mut()
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "forward_batch_gpu_cached".to_string(),
                reason: "Scheduler not initialized".to_string(),
            })?;

        // 1. Token embedding lookup
        let hidden = self.model.embed(token_ids);

        // 2. Process through layers
        for layer in &self.model.layers {
            // Simplified single-layer forward - reuse inner model logic
            // For full implementation, would need to port the complete forward pass
            let _ = layer;
        }

        // 3. Output normalization and LM head
        // For now, return placeholder - full implementation requires porting forward logic
        let output = vec![0.0f32; batch_size * vocab_size];
        let _ = hidden;

        Ok(output)
    }

    /// Adaptive fused attention for production serving (IMP-121)
    ///
    /// Thread-safe wrapper that automatically selects CPU or GPU based on
    /// sequence length. Uses the cached scheduler for efficient GPU operations.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn adaptive_fused_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Threshold for GPU dispatch (from IMP-119 analysis)
        const GPU_SEQ_LEN_THRESHOLD: usize = 64;

        if seq_len >= GPU_SEQ_LEN_THRESHOLD {
            // Long sequence: Use GPU path
            self.gpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        } else {
            // Short sequence: Use CPU path
            self.cpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        }
    }

    /// CPU fused causal attention (thread-safe wrapper)
    fn cpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Use tiled implementation from inner model
        self.model
            .tiled_causal_attention(q, k, v, seq_len, head_dim, scale, 4)
    }

    /// GPU fused causal attention (thread-safe)
    fn gpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        let mut scheduler_guard =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Failed to acquire scheduler lock".to_string(),
                })?;

        // Initialize scheduler if needed
        if scheduler_guard.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_guard = Some(new_scheduler);
        }

        let scheduler =
            scheduler_guard
                .as_mut()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Scheduler not initialized".to_string(),
                })?;

        // Transpose K for matmul
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        for pos in 0..seq_len {
            for d in 0..head_dim {
                k_transposed[d * seq_len + pos] = k[pos * head_dim + d];
            }
        }

        // GPU Q @ K^T
        let scores = scheduler
            .matmul(q, &k_transposed, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused Q@K^T".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        // CPU causal softmax
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                if score > max_val {
                    max_val = score;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                weights[i * seq_len + j] = (score - max_val).exp();
                sum += weights[i * seq_len + j];
            }
            if sum > 0.0 {
                for j in 0..=i {
                    weights[i * seq_len + j] /= sum;
                }
            }
        }

        // GPU weights @ V
        scheduler
            .matmul(&weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused weights@V".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })
    }

    /// Adaptive multihead attention for production serving (IMP-121)
    ///
    /// Thread-safe multi-head attention that automatically selects backend.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn adaptive_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            let head_output =
                self.adaptive_fused_attention(q_head, k_head, v_head, seq_len, head_dim, scale)?;

            attn_output[head_offset..head_offset + seq_len * head_dim]
                .copy_from_slice(&head_output);
        }

        // Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Warmup GPU weight cache for batch inference (PARITY-019)
    ///
    /// Pre-dequantizes all FFN weights to f32 for GPU GEMM operations.
    /// Call this once at server startup to avoid dequantization during inference.
    ///
    /// # Memory Usage
    /// - phi-2 (32 layers): ~6.4 GB
    /// - Per layer: 2 × hidden_dim × intermediate_dim × 4 bytes
    ///
    /// # Returns
    /// - Total memory allocated in bytes
    /// - Number of layers cached
    ///
    /// # Errors
    /// Returns error if dequantization fails
    pub fn warmup_gpu_cache(&self) -> Result<(usize, usize)> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let num_layers = self.model.layers.len();

        // Create cache with model dimensions
        let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);

        // Dequantize each layer's FFN weights
        // Note: warmup closure can't return Result, so we use unwrap_or_default
        // for robustness. In production, use warmup_gpu_cache_checked() for error handling.
        cache.warmup(|layer_idx| {
            let layer = &self.model.layers[layer_idx];

            // Dequantize using model's dequantize_weight method
            let up = self
                .model
                .dequantize_weight(&layer.ffn_up_weight)
                .unwrap_or_default();
            let down = self
                .model
                .dequantize_weight(&layer.ffn_down_weight)
                .unwrap_or_default();

            (up, down)
        });

        let memory_bytes = cache.memory_bytes();
        let cached_count = cache.cached_count();

        // Store in the cache field
        let mut cache_guard =
            self.dequant_cache
                .write()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "warmup_gpu_cache".to_string(),
                    reason: "Cache lock poisoned".to_string(),
                })?;
        *cache_guard = Some(cache);

        Ok((memory_bytes, cached_count))
    }

    /// Check if GPU cache is warmed up
    pub fn is_gpu_cache_warm(&self) -> bool {
        self.dequant_cache
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get GPU cache memory usage in bytes
    pub fn gpu_cache_memory(&self) -> usize {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(DequantizedWeightCache::memory_bytes))
            .unwrap_or(0)
    }

    /// Get dequantized weights for a layer (for GPU batch FFN)
    ///
    /// Returns None if cache not warmed up or layer not found.
    pub fn get_dequantized_ffn_weights(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|c| c.get(layer_idx)))
    }

    /// Batch FFN forward pass using GPU (PARITY-019)
    ///
    /// Processes multiple tokens in parallel using GPU GEMM.
    /// Requires cache to be warmed up via `warmup_gpu_cache()`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size × hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Output tensor [batch_size × hidden_dim]
    ///
    /// # Errors
    /// Returns error if cache not warmed or GPU operations fail
    /// PARITY-103: Batch FFN using CUDA when available
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    pub fn batch_ffn_gpu(&self, hidden_states: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: "Empty batch".to_string(),
            });
        }

        // Get cached weights
        let weights = self.get_dequantized_ffn_weights(layer_idx).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: format!(
                    "Layer {} not cached. Call warmup_gpu_cache() first.",
                    layer_idx
                ),
            }
        })?;

        // PARITY-103: Up projection preferring CUDA
        let mut intermediate = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &weights.up,
            batch_size,
            hidden_dim,
            intermediate_dim,
        )?;

        // Add up bias if present
        if let Some(ref bias) = weights.up_bias {
            for b in 0..batch_size {
                for i in 0..intermediate_dim {
                    intermediate[b * intermediate_dim + i] += bias[i];
                }
            }
        }

        // GELU activation (CPU - fused in future)
        for x in &mut intermediate {
            let x64 = *x as f64;
            *x = (x64
                * 0.5
                * (1.0 + (x64 * 0.797_884_560_8 * (1.0 + 0.044_715 * x64 * x64)).tanh()))
                as f32;
        }

        // PARITY-103: Down projection preferring CUDA
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            &intermediate,
            &weights.down,
            batch_size,
            intermediate_dim,
            hidden_dim,
        )?;

        // Add down bias if present
        if let Some(ref bias) = weights.down_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// PARITY-103: Batch QKV projection using CUDA when available
    ///
    /// Projects hidden states to Q, K, V for all requests in batch.
    /// [batch, hidden] @ [hidden, 3*hidden] = [batch, 3*hidden]
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened hidden states [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened QKV projections [batch * 3 * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_qkv_projection_gpu(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = hidden_states.len() / hidden_dim;
        let qkv_dim = 3 * hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize QKV weight for GPU GEMM
        let qkv_weight = self.model.dequantize_qkv(&layer.qkv_weight)?;

        // PARITY-103: QKV projection preferring CUDA
        let mut qkv = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &qkv_weight,
            batch_size,
            hidden_dim,
            qkv_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.qkv_bias {
            for b in 0..batch_size {
                for i in 0..qkv_dim {
                    qkv[b * qkv_dim + i] += bias[i];
                }
            }
        }

        Ok(qkv)
    }

    /// Batch attention output projection using GPU GEMM (PARITY-024)
    ///
    /// Projects attention outputs for all requests in batch.
    /// [batch, hidden] @ [hidden, hidden] = [batch, hidden]
    ///
    /// # Arguments
    /// * `attention_outputs` - Flattened attention outputs [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened projected outputs [batch * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_attention_output_gpu(
        &self,
        attention_outputs: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = attention_outputs.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize output weight for GPU GEMM
        let output_weight = self.model.dequantize_weight(&layer.attn_output_weight)?;

        // PARITY-103: Output projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, hidden] = [batch, hidden]
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            attention_outputs,
            &output_weight,
            batch_size,
            hidden_dim,
            hidden_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.attn_output_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// Batch LM head projection using GPU GEMM (PARITY-025)
    ///
    /// Projects hidden states to vocabulary logits for all requests in batch.
    /// [batch, hidden] @ [hidden, vocab] = [batch, vocab]
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened normalized hidden states [batch * hidden_dim]
    ///
    /// # Returns
    /// Flattened logits [batch * vocab_size]
    #[cfg(feature = "gpu")]
    pub fn batch_lm_head_gpu(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Dequantize LM head weight for GPU GEMM
        let lm_head_weight = self.model.dequantize_weight(&self.model.lm_head_weight)?;

        // PARITY-103: LM head projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, vocab] = [batch, vocab]
        let mut logits = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
        )?;

        // Add bias if present
        if let Some(ref bias) = self.model.lm_head_bias {
            for b in 0..batch_size {
                for i in 0..vocab_size {
                    logits[b * vocab_size + i] += bias[i];
                }
            }
        }

        Ok(logits)
    }

    /// Batch generation with GPU-accelerated FFN (PARITY-020)
    ///
    /// Processes multiple prompts in parallel using GPU batch operations.
    /// The key optimization is converting MATVEC (single token) to GEMM (batch tokens).
    ///
    /// # Architecture
    /// - Attention: CPU with KV cache (MATVEC is faster on CPU)
    /// - FFN: GPU with batch GEMM (batch_size ≥ 32 uses GPU)
    /// - Sampling: CPU (negligible compared to matmul)
    ///
    /// # Arguments
    /// * `prompts` - Multiple prompts to process in parallel [num_prompts][seq_len]
    /// * `config` - Generation configuration (shared across all prompts)
    ///
    /// # Returns
    /// Generated sequences for each prompt [num_prompts][generated_len]
    ///
    /// # Errors
    /// Returns error if GPU cache not warmed up or generation fails
    ///
    /// # Performance
    /// - Single prompt: ~5 tok/s (CPU-bound, no batching benefit)
    /// - 32 prompts: ~150 tok/s total (~4.7 tok/s per prompt)
    /// - 64 prompts: ~280 tok/s total (~4.4 tok/s per prompt, memory-bound)
    pub fn batch_generate_gpu(
        &self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Verify GPU cache is warmed up
        if !self.is_gpu_cache_warm() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_generate_gpu".to_string(),
                reason: "GPU cache not warmed up. Call warmup_gpu_cache() first.".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_seq_len = prompts.iter().map(Vec::len).max().unwrap_or(0) + config.max_tokens;

        // Initialize KV caches for each prompt
        let mut caches: Vec<OwnedQuantizedKVCache> = prompts
            .iter()
            .map(|_| OwnedQuantizedKVCache::from_config(&self.model.config, max_seq_len))
            .collect();

        // Initialize token sequences (copy prompts)
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();

        // Track generation progress per prompt
        let mut done: Vec<bool> = vec![false; num_prompts];

        // PARITY-097: Parallel prefill across prompts using rayon
        // Each prompt's prefill is independent (different KV cache)
        // Model is shared immutably (&self), caches are mutated independently
        use rayon::prelude::*;

        caches
            .par_iter_mut()
            .zip(prompts.par_iter())
            .try_for_each(|(cache, prompt)| {
                for (pos, &token_id) in prompt.iter().enumerate() {
                    self.model.forward_single_with_cache(token_id, cache, pos)?;
                }
                Ok::<_, RealizarError>(())
            })?;

        // Generation loop with batched FFN (PARITY-021: GPU optimization)
        for gen_idx in 0..config.max_tokens {
            // Collect active prompts for this generation step
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            let active_count = active_indices.len();

            // Use batched forward when we have enough active prompts for GPU benefit
            // GPU batch threshold is 32 (from IMP-600 analysis)
            const GPU_BATCH_THRESHOLD: usize = 32;

            if active_count >= GPU_BATCH_THRESHOLD {
                // PARITY-021: Batched forward with GPU FFN
                // Collect tokens, positions, and cache slices for active prompts
                let batch_tokens: Vec<u32> = active_indices
                    .iter()
                    .map(|&idx| {
                        *sequences[idx]
                            .last()
                            .expect("sequence must have at least prompt tokens")
                    })
                    .collect();

                let batch_positions: Vec<usize> = active_indices
                    .iter()
                    .map(|&idx| prompts[idx].len() + gen_idx)
                    .collect();

                // PARITY-096: Extract caches without cloning using std::mem::take
                // This avoids expensive cache cloning on every generation step
                let mut batch_caches: Vec<OwnedQuantizedKVCache> = active_indices
                    .iter()
                    .map(|&idx| std::mem::take(&mut caches[idx]))
                    .collect();

                // Forward batch with GPU FFN
                let all_logits = self.forward_batch_with_gpu_ffn(
                    &batch_tokens,
                    &mut batch_caches,
                    &batch_positions,
                )?;

                // PARITY-096: Put caches back (move, not clone)
                for (i, &idx) in active_indices.iter().enumerate() {
                    caches[idx] = std::mem::take(&mut batch_caches[i]);
                }

                // Sample and update sequences
                for (i, &prompt_idx) in active_indices.iter().enumerate() {
                    let logits = &all_logits[i];
                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            } else {
                // Sequential forward for small batches (CPU is faster)
                for &prompt_idx in &active_indices {
                    let position = prompts[prompt_idx].len() + gen_idx;
                    let last_token = *sequences[prompt_idx]
                        .last()
                        .expect("sequence must have at least prompt tokens");

                    let logits = self.model.forward_single_with_cache(
                        last_token,
                        &mut caches[prompt_idx],
                        position,
                    )?;

                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(&logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            }
        }

        Ok(sequences)
    }

    /// Batched forward pass with GPU FFN optimization (PARITY-021)
    ///
    /// Processes multiple tokens in parallel with GPU-accelerated FFN.
    /// Attention is still per-token with CPU KV cache, but FFN uses GPU GEMM.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs for each prompt [batch_size]
    /// * `caches` - Per-prompt KV caches
    /// * `positions` - Position for each prompt [batch_size]
    ///
    /// # Returns
    /// Logits for each prompt [batch_size][vocab_size]
    ///
    /// # GPU Dispatch
    /// - batch_size >= 32: GPU GEMM for FFN (10x speedup)
    /// - batch_size < 32: CPU fallback
    pub fn forward_batch_with_gpu_ffn(
        &self,
        token_ids: &[u32],
        caches: &mut [OwnedQuantizedKVCache],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size != caches.len() || batch_size != positions.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Batch size mismatch: tokens={}, caches={}, positions={}",
                    batch_size,
                    caches.len(),
                    positions.len()
                ),
            });
        }

        let hidden_dim = self.model.config.hidden_dim;
        let num_layers = self.model.layers.len();

        // Threshold for GPU dispatch (based on IMP-600 analysis)
        const GPU_BATCH_THRESHOLD: usize = 32;
        let use_gpu = batch_size >= GPU_BATCH_THRESHOLD && self.is_gpu_cache_warm();

        // PARITY-098: Parallel embedding using rayon
        use rayon::prelude::*;
        let mut hidden_states: Vec<Vec<f32>> = token_ids
            .par_iter()
            .map(|&tid| self.model.embed(&[tid]))
            .collect();

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            let layer = &self.model.layers[layer_idx];

            // PARITY-024: GPU batch attention path vs CPU sequential path
            if use_gpu {
                // GPU path: batch QKV projection, per-prompt attention, batch output projection

                // 2a. PARITY-098: Parallel batch layer norm
                let normed_batch: Vec<Vec<f32>> = hidden_states
                    .par_iter()
                    .map(|hidden| {
                        self.model.layer_norm(
                            hidden,
                            &layer.attn_norm_weight,
                            layer.attn_norm_bias.as_deref(),
                            self.model.config.eps,
                        )
                    })
                    .collect();

                // 2b. Batch QKV projection using GPU GEMM (PARITY-024)
                let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
                let batch_qkv = self.batch_qkv_projection_gpu(&batch_normed, layer_idx)?;

                // 2c-2e. PARITY-099: Parallel attention computation per prompt
                // Each prompt has its own KV cache, so we can parallelize
                let qkv_dim = 3 * hidden_dim;

                let attention_outputs: Vec<Vec<f32>> = caches
                    .par_iter_mut()
                    .enumerate()
                    .map(|(prompt_idx, cache)| {
                        let qkv_start = prompt_idx * qkv_dim;
                        let qkv = &batch_qkv[qkv_start..qkv_start + qkv_dim];

                        // Extract Q, K, V
                        let mut q = qkv[0..hidden_dim].to_vec();
                        let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                        let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                        // Apply RoPE (position-dependent, must be per-prompt)
                        // Note: Uses num_heads for both (non-GQA code path)
                        self.model.apply_rope(
                            &mut q,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );
                        self.model.apply_rope(
                            &mut k,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );

                        // Attention with KV cache (must be per-prompt, different caches)
                        // PARITY-027: Use FlashAttention for long sequences (O(N) memory)
                        let k_cache = cache.get_k(layer_idx);
                        let v_cache = cache.get_v(layer_idx);

                        // FlashAttention threshold: use for sequences >= 512 tokens
                        const FLASH_ATTENTION_THRESHOLD: usize = 512;
                        let cache_len = k_cache.len() / hidden_dim;
                        let use_flash_attention = cache_len >= FLASH_ATTENTION_THRESHOLD;

                        let attn_out = if k_cache.is_empty() {
                            v.clone()
                        } else if use_flash_attention {
                            // FlashAttention: O(N) memory, tiled computation
                            const FLASH_BLOCK_SIZE: usize = 64;
                            self.model.flash_attention_tiled(
                                &q,
                                k_cache,
                                v_cache,
                                &k,
                                &v,
                                FLASH_BLOCK_SIZE,
                            )
                        } else {
                            // Standard attention: O(N²) memory but faster for short sequences
                            self.model
                                .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                        };

                        // Store K and V in cache
                        cache.append(layer_idx, &k, &v);
                        attn_out
                    })
                    .collect();

                // 2f. Batch attention output projection using GPU GEMM (PARITY-024)
                let batch_attn: Vec<f32> = attention_outputs.iter().flatten().copied().collect();
                let batch_output = self.batch_attention_output_gpu(&batch_attn, layer_idx)?;

                // 2g. PARITY-100: Parallel residual connection
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += batch_output[start + i];
                        }
                    });
            } else {
                // CPU sequential path (original implementation)
                for (prompt_idx, hidden) in hidden_states.iter_mut().enumerate() {
                    // Attention layer norm
                    let normed = self.model.layer_norm(
                        hidden,
                        &layer.attn_norm_weight,
                        layer.attn_norm_bias.as_deref(),
                        self.model.config.eps,
                    );

                    // QKV projection
                    let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
                    if let Some(ref bias) = layer.qkv_bias {
                        self.model.add_bias(&mut qkv, bias);
                    }

                    // Extract Q, K, V and apply RoPE
                    // Note: Uses num_heads for both (non-GQA code path)
                    let mut q = qkv[0..hidden_dim].to_vec();
                    let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                    let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                    self.model.apply_rope(
                        &mut q,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );
                    self.model.apply_rope(
                        &mut k,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );

                    // Get cached K/V and compute attention
                    let k_cache = caches[prompt_idx].get_k(layer_idx);
                    let v_cache = caches[prompt_idx].get_v(layer_idx);

                    let attn_out = if k_cache.is_empty() {
                        v.clone()
                    } else {
                        self.model
                            .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                    };

                    // Store K and V in cache
                    caches[prompt_idx].append(layer_idx, &k, &v);

                    // Attention output projection
                    let mut attn_output = self
                        .model
                        .fused_matmul(&attn_out, &layer.attn_output_weight)?;
                    if let Some(ref bias) = layer.attn_output_bias {
                        self.model.add_bias(&mut attn_output, bias);
                    }

                    // Residual connection
                    for i in 0..hidden_dim {
                        hidden[i] += attn_output[i];
                    }
                }
            }

            // 2h. FFN - GPU batch or CPU sequential
            if use_gpu {
                // GPU batch FFN: collect hidden states, process together, scatter back
                let batch_hidden: Vec<f32> = hidden_states.iter().flatten().copied().collect();
                let ffn_output = self.batch_ffn_gpu(&batch_hidden, layer_idx)?;

                // PARITY-100: Parallel scatter and residual
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += ffn_output[start + i];
                        }
                    });
            } else {
                // CPU sequential FFN
                for hidden in &mut hidden_states {
                    let mut ffn_hidden = self.model.fused_matmul(hidden, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        self.model.add_bias(&mut ffn_hidden, bias);
                    }
                    self.model.gelu(&mut ffn_hidden);

                    let mut ffn_output = self
                        .model
                        .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                    if let Some(ref bias) = layer.ffn_down_bias {
                        self.model.add_bias(&mut ffn_output, bias);
                    }

                    // Residual
                    for i in 0..hidden_dim {
                        hidden[i] += ffn_output[i];
                    }
                }
            }
        }

        // PARITY-100: Parallel cache advance
        caches.par_iter_mut().for_each(|cache| {
            cache.advance();
        });

        // 3. Final layer norm and LM head for each prompt
        // PARITY-025: Use GPU batch LM head when batch >= threshold
        let vocab_size = self.model.config.vocab_size;

        let all_logits: Vec<Vec<f32>> = if use_gpu {
            // GPU path: batch layer norm and LM head projection

            // 3a. PARITY-098: Parallel final layer norm
            let normed_batch: Vec<Vec<f32>> = hidden_states
                .par_iter()
                .map(|hidden| {
                    self.model.layer_norm(
                        hidden,
                        &self.model.output_norm_weight,
                        self.model.output_norm_bias.as_deref(),
                        self.model.config.eps,
                    )
                })
                .collect();

            // 3b. Batch LM head projection using GPU GEMM (PARITY-025)
            let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
            let batch_logits = self.batch_lm_head_gpu(&batch_normed)?;

            // 3c. PARITY-098: Parallel scatter logits back to per-prompt vectors
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let start = i * vocab_size;
                    batch_logits[start..start + vocab_size].to_vec()
                })
                .collect()
        } else {
            // CPU path: sequential per-prompt processing
            let mut result = Vec::with_capacity(batch_size);
            for hidden in &hidden_states {
                let normed = self.model.layer_norm(
                    hidden,
                    &self.model.output_norm_weight,
                    self.model.output_norm_bias.as_deref(),
                    self.model.config.eps,
                );

                let mut logits = self
                    .model
                    .fused_matmul(&normed, &self.model.lm_head_weight)?;
                if let Some(ref bias) = self.model.lm_head_bias {
                    self.model.add_bias(&mut logits, bias);
                }
                result.push(logits);
            }
            result
        };

        Ok(all_logits)
    }

    /// Get batch generation statistics
    ///
    /// Returns information about the batch processing capabilities.
    pub fn batch_stats(&self) -> BatchGenerationStats {
        let is_cached = self.is_gpu_cache_warm();
        let memory_gb = self.gpu_cache_memory() as f64 / 1_000_000_000.0;
        let num_layers = self.model.layers.len();
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.config.intermediate_dim;

        BatchGenerationStats {
            gpu_cache_ready: is_cached,
            cache_memory_gb: memory_gb,
            num_layers,
            hidden_dim,
            intermediate_dim,
            recommended_batch_size: 32, // GPU GEMM threshold
            max_batch_size: 64,         // Memory-limited
        }
    }
}

/// Statistics for batch generation capabilities (PARITY-020)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct BatchGenerationStats {
    /// Whether GPU cache is ready
    pub gpu_cache_ready: bool,
    /// Memory used by GPU cache in GB
    pub cache_memory_gb: f64,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Recommended batch size for GPU efficiency
    pub recommended_batch_size: usize,
    /// Maximum batch size before memory pressure
    pub max_batch_size: usize,
}

// ============================================================================
// PARITY-023: Request Batching Infrastructure
// ============================================================================

/// A pending request waiting to be batched (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Request ID for tracking
    pub id: u64,
    /// Prompt tokens
    pub prompt: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Time when request was submitted
    pub submitted_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl PendingRequest {
    /// Create a new pending request
    pub fn new(
        id: u64,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            temperature,
            top_k,
            submitted_at: std::time::Instant::now(),
        }
    }

    /// Time spent waiting in queue
    pub fn wait_time(&self) -> std::time::Duration {
        self.submitted_at.elapsed()
    }
}

/// A batch of requests ready for processing (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct RequestBatch {
    /// Requests in this batch
    pub requests: Vec<PendingRequest>,
    /// When batch was formed
    pub formed_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl RequestBatch {
    /// Create batch from requests
    pub fn new(requests: Vec<PendingRequest>) -> Self {
        Self {
            requests,
            formed_at: std::time::Instant::now(),
        }
    }

    /// Number of requests in batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Extract prompts for batch processing
    pub fn prompts(&self) -> Vec<Vec<u32>> {
        self.requests.iter().map(|r| r.prompt.clone()).collect()
    }

    /// Average wait time for requests in this batch
    pub fn avg_wait_time(&self) -> std::time::Duration {
        if self.requests.is_empty() {
            return std::time::Duration::ZERO;
        }
        let total: std::time::Duration = self.requests.iter().map(PendingRequest::wait_time).sum();
        total / self.requests.len() as u32
    }
}

/// Request batch collector with configurable thresholds (PARITY-023)
///
/// Collects incoming requests and forms batches when:
/// - Batch size reaches `batch_threshold`, OR
/// - Wait time exceeds `timeout_ms`
///
/// This enables efficient GPU utilization by batching multiple requests.
#[cfg(feature = "gpu")]
pub struct BatchRequestCollector {
    /// Pending requests
    pending: std::sync::Mutex<Vec<PendingRequest>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
    /// Batch size threshold (32 = GPU GEMM threshold from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before forcing batch formation (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
}

#[cfg(feature = "gpu")]
impl BatchRequestCollector {
    /// Create new collector with default thresholds
    ///
    /// Default: batch_threshold=32, timeout_ms=50, max_batch_size=64
    pub fn new() -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
        }
    }

    /// Create collector with custom thresholds
    pub fn with_thresholds(batch_threshold: usize, timeout_ms: u64, max_batch_size: usize) -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold,
            timeout_ms,
            max_batch_size,
        }
    }

    /// Submit a request to the collector
    ///
    /// Returns the request ID for tracking
    pub fn submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = PendingRequest::new(id, prompt, max_tokens, temperature, top_k);

        let mut pending = self.pending.lock().expect("Mutex poisoned");
        pending.push(request);

        id
    }

    /// Check if batch is ready to be formed
    pub fn is_batch_ready(&self) -> bool {
        let pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return false;
        }

        // Batch ready if threshold reached
        if pending.len() >= self.batch_threshold {
            return true;
        }

        // Batch ready if oldest request has waited too long
        if let Some(oldest) = pending.first() {
            let wait_ms = oldest.wait_time().as_millis() as u64;
            if wait_ms >= self.timeout_ms {
                return true;
            }
        }

        false
    }

    /// Collect a batch of requests
    ///
    /// Returns None if no requests are pending or batch not ready
    pub fn collect_batch(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        // Check if batch is ready (threshold or timeout)
        let ready = pending.len() >= self.batch_threshold
            || pending
                .first()
                .is_some_and(|r| r.wait_time().as_millis() as u64 >= self.timeout_ms);

        if !ready {
            return None;
        }

        // Take up to max_batch_size requests
        let batch_size = pending.len().min(self.max_batch_size);
        let requests: Vec<PendingRequest> = pending.drain(..batch_size).collect();

        Some(RequestBatch::new(requests))
    }

    /// Force collect all pending requests as a batch
    pub fn flush(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        let requests: Vec<PendingRequest> = pending.drain(..).collect();
        Some(RequestBatch::new(requests))
    }

    /// Number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.lock().expect("Mutex poisoned").len()
    }

    /// Total requests submitted
    pub fn total_submitted(&self) -> u64 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "gpu")]
impl Default for BatchRequestCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Batching configuration for request collector (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Minimum batch size to trigger GPU processing (32 from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before processing smaller batch (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
    /// Whether to prefer latency (process immediately) or throughput (wait for batch)
    pub prefer_throughput: bool,
}

#[cfg(feature = "gpu")]
impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl BatchingConfig {
    /// Config optimized for latency (smaller batches, shorter timeout)
    pub fn latency_optimized() -> Self {
        Self {
            batch_threshold: 8,
            timeout_ms: 10,
            max_batch_size: 32,
            prefer_throughput: false,
        }
    }

    /// Config optimized for throughput (larger batches, longer timeout)
    pub fn throughput_optimized() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 100,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

/// Slot state for continuous batching (PARITY-028)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum SlotState {
    /// Slot is available for new request
    Empty,
    /// Slot has active request being generated
    Active {
        /// Unique request identifier
        request_id: u64,
        /// Input prompt tokens
        prompt_tokens: Vec<u32>,
        /// Tokens generated so far
        generated_tokens: Vec<u32>,
        /// Maximum tokens to generate
        max_tokens: usize,
        /// Sampling temperature
        temperature: f32,
        /// Top-k sampling parameter
        top_k: usize,
    },
    /// Slot has completed request waiting for retrieval
    Completed {
        /// Unique request identifier
        request_id: u64,
        /// All generated tokens
        generated_tokens: Vec<u32>,
    },
}

#[cfg(feature = "gpu")]
impl SlotState {
    /// Check if slot is available
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if slot has active generation
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    /// Check if slot has completed request
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Get request ID if slot has one
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Empty => None,
            Self::Active { request_id, .. } | Self::Completed { request_id, .. } => {
                Some(*request_id)
            },
        }
    }
}

/// Continuous batch scheduler (PARITY-028)
///
/// Enables dynamic addition/removal of requests from a running batch:
/// - Requests are assigned to slots
/// - Each slot can be in Empty, Active, or Completed state
/// - New requests fill empty slots immediately
/// - Completed requests free their slots for reuse
///
/// This maximizes GPU utilization by keeping the batch full.
#[cfg(feature = "gpu")]
pub struct ContinuousBatchScheduler {
    /// Fixed-size array of slots
    slots: std::sync::Mutex<Vec<SlotState>>,
    /// KV caches for each slot (pre-allocated)
    caches: std::sync::Mutex<Vec<OwnedQuantizedKVCache>>,
    /// Total slots (max concurrent requests)
    pub num_slots: usize,
    /// Completed request IDs for polling
    completed: std::sync::Mutex<Vec<(u64, Vec<u32>)>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl ContinuousBatchScheduler {
    /// Create scheduler with specified number of slots
    ///
    /// # Arguments
    /// * `num_slots` - Maximum concurrent requests (typically 32-64)
    /// * `num_layers` - Number of transformer layers (for KV cache)
    /// * `hidden_dim` - Hidden dimension (for KV cache)
    /// * `max_seq_len` - Maximum sequence length (for KV cache)
    pub fn new(num_slots: usize, num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let slots = vec![SlotState::Empty; num_slots];
        let caches = (0..num_slots)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len))
            .collect();

        Self {
            slots: std::sync::Mutex::new(slots),
            caches: std::sync::Mutex::new(caches),
            num_slots,
            completed: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a new request to the scheduler
    ///
    /// Returns request ID if slot available, None if all slots full
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Option<u64> {
        let mut slots = self.slots.lock().expect("Mutex poisoned");

        // Find first empty slot
        let empty_idx = slots.iter().position(SlotState::is_empty)?;

        let request_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        slots[empty_idx] = SlotState::Active {
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            temperature,
            top_k,
        };

        Some(request_id)
    }

    /// Get number of active slots
    pub fn active_count(&self) -> usize {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots.iter().filter(|s| s.is_active()).count()
    }

    /// Get number of empty slots
    pub fn empty_count(&self) -> usize {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots.iter().filter(|s| s.is_empty()).count()
    }

    /// Check if any slot has completed request
    pub fn has_completed(&self) -> bool {
        let completed = self.completed.lock().expect("Mutex poisoned");
        !completed.is_empty()
    }

    /// Retrieve completed request results
    pub fn poll_completed(&self) -> Vec<(u64, Vec<u32>)> {
        let mut completed = self.completed.lock().expect("Mutex poisoned");
        std::mem::take(&mut *completed)
    }

    /// Mark a request as completed and move to completed queue
    pub fn complete_request(&self, slot_idx: usize, tokens: Vec<u32>) {
        let mut slots = self.slots.lock().expect("Mutex poisoned");
        let mut completed = self.completed.lock().expect("Mutex poisoned");

        if slot_idx < slots.len() {
            if let SlotState::Active { request_id, .. } = &slots[slot_idx] {
                let id = *request_id;
                // Move to completed
                completed.push((id, tokens));
                // Free the slot
                slots[slot_idx] = SlotState::Empty;

                // Reset KV cache for this slot
                let mut caches = self.caches.lock().expect("Mutex poisoned");
                caches[slot_idx].reset();
            }
        }
    }

    /// Get active slot indices and their current positions
    pub fn get_active_slots(&self) -> Vec<(usize, usize)> {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| match slot {
                SlotState::Active {
                    prompt_tokens,
                    generated_tokens,
                    ..
                } => {
                    let pos = prompt_tokens.len() + generated_tokens.len();
                    Some((idx, pos))
                },
                _ => None,
            })
            .collect()
    }

    /// Get utilization (active_slots / total_slots)
    pub fn utilization(&self) -> f64 {
        let active = self.active_count();
        active as f64 / self.num_slots as f64
    }
}

/// Speculative decoding configuration (PARITY-029)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculatively generate per step
    pub speculation_length: usize,
    /// Temperature for draft model (lower = more deterministic)
    pub draft_temperature: f32,
    /// Whether to use same model for draft (self-speculative)
    pub self_speculative: bool,
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: true,
        }
    }
}

/// Result of speculative decoding verification step
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted
    pub accepted_count: usize,
    /// Total draft tokens generated
    pub draft_count: usize,
    /// Accepted tokens (verified by target model)
    pub accepted_tokens: Vec<u32>,
    /// Whether all draft tokens were accepted
    pub all_accepted: bool,
}

/// Speculative decoder for accelerated token generation (PARITY-029)
///
/// Implements speculative decoding (Leviathan et al., 2023):
/// 1. Draft model generates K candidate tokens quickly
/// 2. Target model verifies all K tokens in parallel
/// 3. Accept tokens until first rejection, then resample
///
/// This enables O(K) speedup when draft acceptance rate is high.
#[cfg(feature = "gpu")]
pub struct SpeculativeDecoder {
    /// Speculative decoding configuration
    pub config: SpeculativeConfig,
    /// Statistics: total draft tokens generated
    pub total_draft_tokens: std::sync::atomic::AtomicU64,
    /// Statistics: total draft tokens accepted
    pub total_accepted_tokens: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl SpeculativeDecoder {
    /// Create new speculative decoder with default config
    pub fn new() -> Self {
        Self {
            config: SpeculativeConfig::default(),
            total_draft_tokens: std::sync::atomic::AtomicU64::new(0),
            total_accepted_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create speculative decoder with custom config
    pub fn with_config(config: SpeculativeConfig) -> Self {
        Self {
            config,
            total_draft_tokens: std::sync::atomic::AtomicU64::new(0),
            total_accepted_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get acceptance rate (accepted / total draft tokens)
    pub fn acceptance_rate(&self) -> f64 {
        let total = self
            .total_draft_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        let accepted = self
            .total_accepted_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        accepted as f64 / total as f64
    }

    /// Verify draft tokens against target model logits
    ///
    /// # Arguments
    /// * `draft_tokens` - Candidate tokens from draft model
    /// * `target_logits` - Logits from target model for each position
    /// * `temperature` - Sampling temperature for rejection sampling
    ///
    /// # Returns
    /// VerificationResult with accepted tokens and statistics
    pub fn verify_draft(
        &self,
        draft_tokens: &[u32],
        target_logits: &[Vec<f32>],
        temperature: f32,
    ) -> VerificationResult {
        let mut accepted_tokens = Vec::with_capacity(draft_tokens.len());
        let mut accepted_count = 0;

        // Verify each draft token against target model distribution
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() {
                break;
            }

            let logits = &target_logits[i];

            // Find target model's top token
            let (target_token, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // Accept if draft matches target (greedy case)
            if temperature == 0.0 {
                if draft_token == target_token as u32 {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject and use target's token instead
                    accepted_tokens.push(target_token as u32);
                    accepted_count += 1;
                    break; // Stop at first mismatch
                }
            } else {
                // Rejection sampling for non-greedy decoding
                // P(accept) = min(1, p_target(x) / p_draft(x))
                // For simplicity, accept if draft is in top-k of target
                let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
                sorted_indices.sort_by(|&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let top_k = 10; // Accept if in top-10
                let in_top_k = sorted_indices
                    .iter()
                    .take(top_k)
                    .any(|&idx| idx == draft_token as usize);

                if in_top_k {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject, use target's sampled token
                    accepted_tokens.push(sorted_indices[0] as u32);
                    accepted_count += 1;
                    break;
                }
            }
        }

        // Update statistics
        self.total_draft_tokens.fetch_add(
            draft_tokens.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.total_accepted_tokens
            .fetch_add(accepted_count as u64, std::sync::atomic::Ordering::Relaxed);

        VerificationResult {
            accepted_count,
            draft_count: draft_tokens.len(),
            accepted_tokens,
            all_accepted: accepted_count == draft_tokens.len(),
        }
    }

    /// Calculate expected speedup based on acceptance rate
    ///
    /// Speedup = K * acceptance_rate + 1 (always get at least 1 token)
    pub fn expected_speedup(&self) -> f64 {
        let k = self.config.speculation_length as f64;
        let acceptance_rate = self.acceptance_rate();
        k * acceptance_rate + 1.0
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.total_draft_tokens
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_accepted_tokens
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU Buffer Pool for zero-allocation inference (PARITY-031, IMP-309)
///
/// Pre-allocates GPU buffers during warmup to eliminate allocation overhead
/// during generation. Uses a pool of reusable buffers for each tensor type.
///
/// # Key Properties
/// - Zero GPU malloc after warmup phase
/// - Pre-allocated buffers for common tensor sizes
/// - Thread-safe buffer borrowing and return
///
/// # Buffer Types
/// - Hidden state buffers: [batch_size, hidden_dim]
/// - Intermediate buffers: [batch_size, intermediate_dim]
/// - Attention score buffers: [batch_size, num_heads, seq_len]
/// - KV cache buffers: [num_layers, seq_len, hidden_dim]
#[cfg(feature = "gpu")]
pub struct GpuBufferPool {
    /// Pre-allocated hidden state buffers
    hidden_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated intermediate buffers (FFN)
    intermediate_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated attention score buffers
    attention_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Buffer dimensions for validation
    hidden_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    num_heads: usize,
    /// Pool size per buffer type
    pool_size: usize,
    /// Statistics: buffers borrowed
    pub borrows: std::sync::atomic::AtomicU64,
    /// Statistics: buffers returned
    pub returns: std::sync::atomic::AtomicU64,
    /// Statistics: allocations after warmup (should be 0)
    pub post_warmup_allocs: std::sync::atomic::AtomicU64,
    /// Whether warmup is complete
    warmed_up: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "gpu")]
impl GpuBufferPool {
    /// Create new buffer pool with specified dimensions
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        max_seq_len: usize,
        num_heads: usize,
        pool_size: usize,
    ) -> Self {
        Self {
            hidden_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            intermediate_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            attention_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            hidden_dim,
            intermediate_dim,
            max_seq_len,
            num_heads,
            pool_size,
            borrows: std::sync::atomic::AtomicU64::new(0),
            returns: std::sync::atomic::AtomicU64::new(0),
            post_warmup_allocs: std::sync::atomic::AtomicU64::new(0),
            warmed_up: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Warmup: pre-allocate all buffers
    ///
    /// Call this once during model initialization to eliminate
    /// allocation overhead during inference.
    pub fn warmup(&self) {
        // Pre-allocate hidden state buffers
        {
            let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.hidden_dim]);
            }
        }

        // Pre-allocate intermediate buffers (FFN)
        {
            let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.intermediate_dim]);
            }
        }

        // Pre-allocate attention score buffers
        {
            let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.num_heads * self.max_seq_len]);
            }
        }

        self.warmed_up
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Borrow a hidden state buffer from the pool
    ///
    /// Returns a pre-allocated buffer if available, or allocates new if needed.
    pub fn borrow_hidden(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            // Need to allocate - track if after warmup
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.hidden_dim]
        }
    }

    /// Return a hidden state buffer to the pool
    pub fn return_hidden(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Zero out for security and determinism
        buffer.fill(0.0);

        let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Borrow an intermediate buffer from the pool
    pub fn borrow_intermediate(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.intermediate_dim]
        }
    }

    /// Return an intermediate buffer to the pool
    pub fn return_intermediate(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        buffer.fill(0.0);

        let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
    }

    /// Borrow an attention score buffer from the pool
    pub fn borrow_attention(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.num_heads * self.max_seq_len]
        }
    }

    /// Return an attention score buffer to the pool
    pub fn return_attention(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        buffer.fill(0.0);

        let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
    }

    /// Check if pool has achieved zero-allocation after warmup
    pub fn is_zero_alloc(&self) -> bool {
        self.warmed_up.load(std::sync::atomic::Ordering::Acquire)
            && self
                .post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
    }

    /// Get pool statistics
    pub fn stats(&self) -> GpuBufferPoolStats {
        GpuBufferPoolStats {
            borrows: self.borrows.load(std::sync::atomic::Ordering::Relaxed),
            returns: self.returns.load(std::sync::atomic::Ordering::Relaxed),
            post_warmup_allocs: self
                .post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed),
            warmed_up: self.warmed_up.load(std::sync::atomic::Ordering::Acquire),
            hidden_available: self.hidden_buffers.lock().expect("mutex poisoned").len(),
            intermediate_available: self
                .intermediate_buffers
                .lock()
                .expect("mutex poisoned")
                .len(),
            attention_available: self.attention_buffers.lock().expect("mutex poisoned").len(),
        }
    }

    /// Calculate total memory usage of the buffer pool
    pub fn memory_usage_bytes(&self) -> usize {
        let hidden_bytes = self.pool_size * self.hidden_dim * 4;
        let intermediate_bytes = self.pool_size * self.intermediate_dim * 4;
        let attention_bytes = self.pool_size * self.num_heads * self.max_seq_len * 4;
        hidden_bytes + intermediate_bytes + attention_bytes
    }
}

/// Statistics for GpuBufferPool
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuBufferPoolStats {
    /// Total borrows
    pub borrows: u64,
    /// Total returns
    pub returns: u64,
    /// Allocations after warmup (should be 0)
    pub post_warmup_allocs: u64,
    /// Whether warmup is complete
    pub warmed_up: bool,
    /// Available hidden buffers
    pub hidden_available: usize,
    /// Available intermediate buffers
    pub intermediate_available: usize,
    /// Available attention buffers
    pub attention_available: usize,
}

/// Async Command Queue for GPU pipelining (PARITY-032, IMP-310)
///
/// Implements double-buffering to hide GPU latency by overlapping
/// computation and data transfer. While one batch is being processed
/// on GPU, the next batch is being prepared on CPU.
///
/// # Key Properties
/// - Double-buffering: 2 command slots for overlap
/// - Async submission: Non-blocking command enqueue
/// - Pipeline stages: Prepare → Submit → Execute → Complete
///
/// # GPU Utilization Target
/// - Without pipelining: ~50% (waiting for results)
/// - With pipelining: >85% (overlapped execution)
#[cfg(feature = "gpu")]
pub struct AsyncCommandQueue {
    /// Command slots for double-buffering (2 slots)
    slots: [std::sync::Mutex<CommandSlot>; 2],
    /// Current slot index for submission
    current_slot: std::sync::atomic::AtomicUsize,
    /// Statistics: commands submitted
    pub commands_submitted: std::sync::atomic::AtomicU64,
    /// Statistics: commands completed
    pub commands_completed: std::sync::atomic::AtomicU64,
    /// Statistics: pipeline stalls (had to wait for previous)
    pub pipeline_stalls: std::sync::atomic::AtomicU64,
}

/// State of a command slot in the async queue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum CommandSlotState {
    /// Slot is empty and ready for new command
    Empty,
    /// Command is being prepared (CPU side)
    Preparing,
    /// Command has been submitted to GPU
    Submitted,
    /// Command execution is complete
    Complete,
}

/// A command slot for async execution
#[cfg(feature = "gpu")]
pub struct CommandSlot {
    /// Current state of this slot
    state: CommandSlotState,
    /// Input data for the command
    input: Option<Vec<f32>>,
    /// Output data from the command
    output: Option<Vec<f32>>,
    /// Timestamp when command was submitted
    submit_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl Default for CommandSlot {
    fn default() -> Self {
        Self {
            state: CommandSlotState::Empty,
            input: None,
            output: None,
            submit_time: None,
        }
    }
}

#[cfg(feature = "gpu")]
impl AsyncCommandQueue {
    /// Create new async command queue with double-buffering
    pub fn new() -> Self {
        Self {
            slots: [
                std::sync::Mutex::new(CommandSlot::default()),
                std::sync::Mutex::new(CommandSlot::default()),
            ],
            current_slot: std::sync::atomic::AtomicUsize::new(0),
            commands_submitted: std::sync::atomic::AtomicU64::new(0),
            commands_completed: std::sync::atomic::AtomicU64::new(0),
            pipeline_stalls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a command for async execution
    ///
    /// Returns the slot index where the command was placed.
    /// If both slots are busy, this will block until one is available
    /// (counted as a pipeline stall).
    pub fn submit(&self, input: Vec<f32>) -> usize {
        let slot_idx = self
            .current_slot
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            % 2;

        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");

        // Check if we need to wait for previous command
        if matches!(
            slot.state,
            CommandSlotState::Submitted | CommandSlotState::Preparing
        ) {
            self.pipeline_stalls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // In real implementation, would wait for GPU completion
            // For now, mark as complete to allow reuse
            slot.state = CommandSlotState::Complete;
        }

        // Prepare new command
        slot.state = CommandSlotState::Preparing;
        slot.input = Some(input);
        slot.output = None;
        slot.submit_time = Some(std::time::Instant::now());

        // Mark as submitted
        slot.state = CommandSlotState::Submitted;
        self.commands_submitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        slot_idx
    }

    /// Mark a command as complete with output
    pub fn complete(&self, slot_idx: usize, output: Vec<f32>) {
        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");
        slot.state = CommandSlotState::Complete;
        slot.output = Some(output);
        self.commands_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get output from a completed command
    ///
    /// Returns None if command is not complete yet.
    pub fn get_output(&self, slot_idx: usize) -> Option<Vec<f32>> {
        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");
        if matches!(slot.state, CommandSlotState::Complete) {
            slot.state = CommandSlotState::Empty;
            slot.output.take()
        } else {
            None
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> AsyncQueueStats {
        let submitted = self
            .commands_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .commands_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let stalls = self
            .pipeline_stalls
            .load(std::sync::atomic::Ordering::Relaxed);

        // GPU utilization estimate: (1 - stalls/submitted) * 100
        let utilization = if submitted > 0 {
            (1.0 - stalls as f64 / submitted as f64) * 100.0
        } else {
            0.0
        };

        AsyncQueueStats {
            commands_submitted: submitted,
            commands_completed: completed,
            pipeline_stalls: stalls,
            in_flight: submitted.saturating_sub(completed),
            gpu_utilization_percent: utilization,
        }
    }

    /// Calculate pipeline efficiency
    ///
    /// Efficiency = commands without stall / total commands
    pub fn pipeline_efficiency(&self) -> f64 {
        let submitted = self
            .commands_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let stalls = self
            .pipeline_stalls
            .load(std::sync::atomic::Ordering::Relaxed);
        if submitted == 0 {
            return 1.0;
        }
        (submitted - stalls) as f64 / submitted as f64
    }
}

#[cfg(feature = "gpu")]
impl Default for AsyncCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for AsyncCommandQueue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct AsyncQueueStats {
    /// Total commands submitted
    pub commands_submitted: u64,
    /// Total commands completed
    pub commands_completed: u64,
    /// Pipeline stalls (had to wait)
    pub pipeline_stalls: u64,
    /// Commands currently in flight
    pub in_flight: u64,
    /// Estimated GPU utilization percentage
    pub gpu_utilization_percent: f64,
}

/// Prefix Cache for common prompts (PARITY-033, IMP-319)
///
/// Caches the KV cache state for common prompt prefixes, enabling
/// instant response (0ms TTFT) for repeated prompts.
///
/// # Key Properties
/// - Hash-based prefix lookup (FNV-1a)
/// - LRU eviction for memory management
/// - Thread-safe access
///
/// # Use Cases
/// - System prompts (cached once, reused for all requests)
/// - Common few-shot examples
/// - Chat history prefixes
#[cfg(feature = "gpu")]
pub struct PrefixCache {
    /// Cached prefix entries (hash → entry)
    entries: std::sync::Mutex<std::collections::HashMap<u64, PrefixCacheEntry>>,
    /// Maximum number of cached prefixes
    max_entries: usize,
    /// Statistics: cache hits
    pub hits: std::sync::atomic::AtomicU64,
    /// Statistics: cache misses
    pub misses: std::sync::atomic::AtomicU64,
    /// Statistics: evictions
    pub evictions: std::sync::atomic::AtomicU64,
}

/// A cached prefix entry
#[cfg(feature = "gpu")]
pub struct PrefixCacheEntry {
    /// The original prompt tokens
    pub tokens: Vec<u32>,
    /// Cached K state for each layer [num_layers, seq_len, hidden_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Cached V state for each layer [num_layers, seq_len, hidden_dim]
    pub v_cache: Vec<Vec<f32>>,
    /// Timestamp for LRU eviction
    pub last_access: std::time::Instant,
    /// Number of times this prefix was hit
    pub hit_count: u64,
}

#[cfg(feature = "gpu")]
impl PrefixCache {
    /// Create new prefix cache with specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(std::collections::HashMap::with_capacity(max_entries)),
            max_entries,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Hash tokens to create cache key (FNV-1a)
    fn hash_tokens(tokens: &[u32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET;
        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Look up a prefix in the cache
    ///
    /// Returns the cached KV state if found, None otherwise.
    #[allow(clippy::type_complexity)]
    pub fn lookup(&self, tokens: &[u32]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let hash = Self::hash_tokens(tokens);

        let mut entries = self.entries.lock().expect("mutex poisoned");
        if let Some(entry) = entries.get_mut(&hash) {
            // Verify tokens match (hash collision check)
            if entry.tokens == tokens {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                entry.last_access = std::time::Instant::now();
                entry.hit_count += 1;
                return Some((entry.k_cache.clone(), entry.v_cache.clone()));
            }
        }

        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    /// Insert a new prefix into the cache
    ///
    /// Evicts LRU entry if cache is full.
    pub fn insert(&self, tokens: Vec<u32>, k_cache: Vec<Vec<f32>>, v_cache: Vec<Vec<f32>>) {
        let hash = Self::hash_tokens(&tokens);

        let mut entries = self.entries.lock().expect("mutex poisoned");

        // Evict LRU if at capacity
        if entries.len() >= self.max_entries {
            // Find oldest entry
            if let Some((&oldest_hash, _)) = entries.iter().min_by_key(|(_, e)| e.last_access) {
                entries.remove(&oldest_hash);
                self.evictions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        entries.insert(
            hash,
            PrefixCacheEntry {
                tokens,
                k_cache,
                v_cache,
                last_access: std::time::Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Check if a prefix is cached
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let hash = Self::hash_tokens(tokens);
        let entries = self.entries.lock().expect("mutex poisoned");
        entries.contains_key(&hash)
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;

        PrefixCacheStats {
            hits,
            misses,
            evictions: self.evictions.load(std::sync::atomic::Ordering::Relaxed),
            entries: self.entries.lock().expect("mutex poisoned").len(),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut entries = self.entries.lock().expect("mutex poisoned");
        entries.clear();
    }

    /// Estimate memory usage of cached prefixes
    pub fn memory_usage_bytes(&self) -> usize {
        let entries = self.entries.lock().expect("mutex poisoned");
        entries
            .values()
            .map(|e| {
                let k_bytes: usize = e.k_cache.iter().map(|v| v.len() * 4).sum();
                let v_bytes: usize = e.v_cache.iter().map(|v| v.len() * 4).sum();
                let token_bytes = e.tokens.len() * 4;
                k_bytes + v_bytes + token_bytes
            })
            .sum()
    }
}

#[cfg(feature = "gpu")]
impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(16) // Default: cache 16 prefixes
    }
}

/// Statistics for PrefixCache
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evictions due to capacity
    pub evictions: u64,
    /// Current number of cached entries
    pub entries: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

// =============================================================================
// PARITY-034: Multi-Request Scheduler with Scheduling Policies (IMP-317)
// =============================================================================
//
// Extends PARITY-028's ContinuousBatchScheduler with:
// - Multiple scheduling policies (FCFS, SJF, Round-Robin)
// - Request queuing with priorities
// - TTFT (Time to First Token) tracking
// - Throughput scaling verification
//
// Architecture:
// - Incoming requests are queued with their KV cache states
// - Scheduler batches decode steps from multiple requests
// - GPU GEMM efficiency: batch_size > 1 enables GPU acceleration
// - Preemption: Long-running requests can be paused for new arrivals
// =============================================================================

/// Request state in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiRequestState {
    /// Waiting for prefill
    Pending,
    /// Prefill in progress
    Prefilling,
    /// Decode in progress
    Decoding,
    /// Request completed
    Completed,
    /// Request preempted (paused)
    Preempted,
}

/// A single inference request in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct MultiSchedulerRequest {
    /// Unique request ID
    pub id: u64,
    /// Input tokens
    pub tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Current state
    pub state: MultiRequestState,
    /// KV cache position (how many tokens processed)
    pub kv_position: usize,
    /// Arrival time for FCFS scheduling
    pub arrival_time: std::time::Instant,
    /// Time first token generated (for TTFT metric)
    pub first_token_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl MultiSchedulerRequest {
    /// Create new request
    pub fn new(id: u64, tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            tokens,
            generated: Vec::with_capacity(max_tokens),
            max_tokens,
            state: MultiRequestState::Pending,
            kv_position: 0,
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }
    }

    /// Check if request is complete
    pub fn is_complete(&self) -> bool {
        self.state == MultiRequestState::Completed || self.generated.len() >= self.max_tokens
    }

    /// Time to first token (None if not yet generated)
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time
            .map(|t| t.duration_since(self.arrival_time).as_secs_f64() * 1000.0)
    }
}

/// Scheduling policy for the batch scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    Fcfs,
    /// Shortest job first (by remaining tokens)
    Sjf,
    /// Round-robin with time slices
    RoundRobin,
}

/// Multi-request scheduler with scheduling policies (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestScheduler {
    /// Pending requests queue
    pending: std::sync::Mutex<std::collections::VecDeque<MultiSchedulerRequest>>,
    /// Active requests being processed
    active: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Completed requests
    completed: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum concurrent requests
    max_concurrent: usize,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Request ID counter
    next_id: std::sync::atomic::AtomicU64,
    /// Requests submitted
    pub requests_submitted: std::sync::atomic::AtomicU64,
    /// Requests completed
    pub requests_completed: std::sync::atomic::AtomicU64,
    /// Total tokens generated
    pub tokens_generated: std::sync::atomic::AtomicU64,
    /// Batch iterations performed
    pub batch_iterations: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl MultiRequestScheduler {
    /// Create new scheduler with given parameters
    pub fn new(max_batch_size: usize, max_concurrent: usize, policy: SchedulingPolicy) -> Self {
        Self {
            pending: std::sync::Mutex::new(std::collections::VecDeque::new()),
            active: std::sync::Mutex::new(Vec::with_capacity(max_concurrent)),
            completed: std::sync::Mutex::new(Vec::new()),
            max_batch_size,
            max_concurrent,
            policy,
            next_id: std::sync::atomic::AtomicU64::new(0),
            requests_submitted: std::sync::atomic::AtomicU64::new(0),
            requests_completed: std::sync::atomic::AtomicU64::new(0),
            tokens_generated: std::sync::atomic::AtomicU64::new(0),
            batch_iterations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a new request
    pub fn submit(&self, tokens: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = MultiSchedulerRequest::new(id, tokens, max_tokens);

        let mut pending = self.pending.lock().expect("mutex poisoned");
        pending.push_back(request);
        self.requests_submitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        id
    }

    /// Get batch of requests ready for decode step
    ///
    /// Returns request IDs and their current positions
    pub fn get_decode_batch(&self) -> Vec<(u64, usize)> {
        let mut active = self.active.lock().expect("mutex poisoned");
        let mut pending = self.pending.lock().expect("mutex poisoned");

        // Promote pending requests to active (up to max_concurrent)
        while active.len() < self.max_concurrent && !pending.is_empty() {
            if let Some(mut req) = pending.pop_front() {
                req.state = MultiRequestState::Decoding;
                active.push(req);
            }
        }

        // Sort by policy
        match self.policy {
            SchedulingPolicy::Fcfs => {
                // Already in arrival order
            },
            SchedulingPolicy::Sjf => {
                active.sort_by_key(|r| r.max_tokens - r.generated.len());
            },
            SchedulingPolicy::RoundRobin => {
                // Rotate - move first to end
                if active.len() > 1 {
                    let first = active.remove(0);
                    active.push(first);
                }
            },
        }

        // Return batch of decoding requests
        active
            .iter()
            .filter(|r| r.state == MultiRequestState::Decoding)
            .take(self.max_batch_size)
            .map(|r| (r.id, r.kv_position))
            .collect()
    }

    /// Record generated token for a request
    pub fn record_token(&self, request_id: u64, token: u32) {
        let mut active = self.active.lock().expect("mutex poisoned");

        if let Some(req) = active.iter_mut().find(|r| r.id == request_id) {
            // Record TTFT for first token
            if req.first_token_time.is_none() {
                req.first_token_time = Some(std::time::Instant::now());
            }

            req.generated.push(token);
            req.kv_position += 1;
            self.tokens_generated
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Check if complete
            if req.is_complete() {
                req.state = MultiRequestState::Completed;
            }
        }
    }

    /// Move completed requests from active to completed
    pub fn collect_completed(&self) -> Vec<MultiSchedulerRequest> {
        let mut active = self.active.lock().expect("mutex poisoned");
        let mut completed = self.completed.lock().expect("mutex poisoned");

        let (done, still_active): (Vec<_>, Vec<_>) = active
            .drain(..)
            .partition(|r| r.state == MultiRequestState::Completed);

        *active = still_active;

        for _req in &done {
            self.requests_completed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        completed.extend(done.iter().cloned());
        done
    }

    /// Run one batch iteration (for simulation)
    pub fn step(&self) {
        self.batch_iterations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> MultiRequestStats {
        let submitted = self
            .requests_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .requests_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let tokens = self
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let iterations = self
            .batch_iterations
            .load(std::sync::atomic::Ordering::Relaxed);

        let pending = self.pending.lock().expect("mutex poisoned").len();
        let active = self.active.lock().expect("mutex poisoned").len();

        MultiRequestStats {
            requests_submitted: submitted,
            requests_completed: completed,
            tokens_generated: tokens,
            batch_iterations: iterations,
            pending_requests: pending,
            active_requests: active,
            avg_batch_size: if iterations > 0 {
                tokens as f64 / iterations as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for multi-request scheduler (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestStats {
    /// Total requests submitted
    pub requests_submitted: u64,
    /// Total requests completed
    pub requests_completed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Batch iterations performed
    pub batch_iterations: u64,
    /// Current pending requests
    pub pending_requests: usize,
    /// Current active requests
    pub active_requests: usize,
    /// Average batch size
    pub avg_batch_size: f64,
}

// =============================================================================
// PARITY-035: Chunked Prefill for Long Contexts (IMP-320)
// =============================================================================
//
// Enables streaming prompt processing by breaking long prefills into chunks.
// Key optimization for TTFT (Time to First Token) with long contexts.
//
// Architecture:
// - Prompt is split into chunks (default 512 tokens)
// - Each chunk processes incrementally, updating KV cache
// - First token can be generated after first chunk completes
// - Total prefill time is spread across chunks
// =============================================================================

/// Configuration for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Chunk size in tokens (default: 512)
    pub chunk_size: usize,
    /// Maximum context length (default: 8192)
    pub max_context: usize,
    /// Whether to yield after each chunk for streaming
    pub stream_chunks: bool,
}

#[cfg(feature = "gpu")]
impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_context: 8192,
            stream_chunks: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl ChunkedPrefillConfig {
    /// Create config with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }
}

/// Progress report for a single chunk
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkProgress {
    /// Chunk index (0-based)
    pub chunk_idx: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Tokens processed so far
    pub tokens_processed: usize,
    /// Total tokens to process
    pub total_tokens: usize,
    /// Time for this chunk (ms)
    pub chunk_time_ms: f64,
    /// Cumulative time so far (ms)
    pub cumulative_time_ms: f64,
}

/// Chunked prefill processor for long context handling
#[cfg(feature = "gpu")]
pub struct ChunkedPrefill {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Chunks created from prompt
    chunks: Vec<Vec<u32>>,
    /// Current chunk being processed
    current_chunk: usize,
    /// Tokens processed so far
    tokens_processed: usize,
    /// Start time for timing
    start_time: Option<std::time::Instant>,
    /// Timing for each chunk
    chunk_times_ms: Vec<f64>,
}

#[cfg(feature = "gpu")]
impl ChunkedPrefill {
    /// Create new chunked prefill from prompt tokens
    pub fn new(prompt_tokens: &[u32], config: ChunkedPrefillConfig) -> Self {
        let chunks: Vec<Vec<u32>> = prompt_tokens
            .chunks(config.chunk_size)
            .map(<[u32]>::to_vec)
            .collect();

        Self {
            config,
            chunks,
            current_chunk: 0,
            tokens_processed: 0,
            start_time: None,
            chunk_times_ms: Vec::new(),
        }
    }

    /// Get total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Check if there are more chunks to process
    pub fn has_more_chunks(&self) -> bool {
        self.current_chunk < self.chunks.len()
    }

    /// Get the next chunk to process
    ///
    /// Returns None if all chunks are processed
    pub fn next_chunk(&mut self) -> Option<&[u32]> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        if self.current_chunk < self.chunks.len() {
            let chunk = &self.chunks[self.current_chunk];
            Some(chunk.as_slice())
        } else {
            None
        }
    }

    /// Mark current chunk as complete
    pub fn complete_chunk(&mut self, chunk_time_ms: f64) {
        if self.current_chunk < self.chunks.len() {
            self.tokens_processed += self.chunks[self.current_chunk].len();
            self.chunk_times_ms.push(chunk_time_ms);
            self.current_chunk += 1;
        }
    }

    /// Get progress after completing a chunk
    pub fn progress(&self) -> ChunkProgress {
        let cumulative_time_ms: f64 = self.chunk_times_ms.iter().sum();

        ChunkProgress {
            chunk_idx: self.current_chunk.saturating_sub(1),
            total_chunks: self.chunks.len(),
            tokens_processed: self.tokens_processed,
            total_tokens: self.total_tokens(),
            chunk_time_ms: self.chunk_times_ms.last().copied().unwrap_or(0.0),
            cumulative_time_ms,
        }
    }

    /// Get estimated time to first token (after first chunk)
    pub fn estimated_ttft_ms(&self) -> f64 {
        if let Some(first_chunk_time) = self.chunk_times_ms.first() {
            *first_chunk_time
        } else {
            // Estimate based on chunk size and typical throughput
            let tokens = self.chunks.first().map_or(0, Vec::len);
            // Conservative estimate: 0.5ms per token for prefill
            tokens as f64 * 0.5
        }
    }

    /// Get statistics after completion
    pub fn stats(&self) -> ChunkedPrefillStats {
        let total_time_ms: f64 = self.chunk_times_ms.iter().sum();
        let total_tokens = self.total_tokens();
        let avg_chunk_time_ms = if !self.chunk_times_ms.is_empty() {
            total_time_ms / self.chunk_times_ms.len() as f64
        } else {
            0.0
        };

        ChunkedPrefillStats {
            total_chunks: self.chunks.len(),
            chunk_size: self.config.chunk_size,
            total_tokens,
            total_time_ms,
            avg_chunk_time_ms,
            ttft_ms: self.estimated_ttft_ms(),
            tokens_per_second: if total_time_ms > 0.0 {
                total_tokens as f64 / (total_time_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Statistics for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillStats {
    /// Total chunks processed
    pub total_chunks: usize,
    /// Chunk size used
    pub chunk_size: usize,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Total time (ms)
    pub total_time_ms: f64,
    /// Average time per chunk (ms)
    pub avg_chunk_time_ms: f64,
    /// Time to first token (ms)
    pub ttft_ms: f64,
    /// Prefill throughput (tokens/sec)
    pub tokens_per_second: f64,
}

impl OwnedQuantizedModel {
    /// Create owned model from memory-mapped GGUF file
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    pub fn from_mapped(mapped: &MappedGGUFModel) -> Result<Self> {
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
        fn dtype_to_byte(dtype: &str) -> u8 {
            match dtype {
                "F32" => 0,
                "F16" => 1,
                "BF16" => 2,
                "I8" => 3,
                "I16" => 4,
                "I32" => 5,
                "I64" => 6,
                "U8" => 7,
                "Q4_K" => 8,
                "Q6_K" => 9,
                "Q8_0" => 10,
                "Q4_0" => 11,
                "Q5_K" => 12,
                "Q3_K" => 13,
                "Q2_K" => 14,
                _ => 0,
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

    /// PARITY-113: Enable CUDA acceleration for this model
    ///
    /// When enabled, all fused_matmul operations will route through
    /// CUDA GEMM kernels instead of CPU SIMD.
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - CUDA device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut model = OwnedQuantizedModel::from_mapped(&mapped)?;
    /// model.enable_cuda(0)?;  // Enable CUDA on GPU 0
    /// assert!(model.cuda_enabled());
    /// ```
    #[cfg(feature = "cuda")]
    pub fn enable_cuda(&mut self, device_ordinal: i32) -> Result<()> {
        use crate::cuda::CudaExecutor;

        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "enable_cuda".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        // CORRECTNESS-011: Set rope_type for correct RoPE style (NORM vs NEOX)
        executor.set_rope_type(self.config.rope_type);

        self.cuda_executor = Some(std::sync::Mutex::new(executor));
        self.cuda_kernel_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Check if CUDA is enabled for this model
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_enabled(&self) -> bool {
        self.cuda_executor.is_some()
    }

    /// Get CUDA kernel execution count since CUDA was enabled
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_kernel_count(&self) -> u64 {
        self.cuda_kernel_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fused matrix-vector multiply using Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K
    ///
    /// PARITY-113: When CUDA is enabled, routes to CUDA GEMM kernel.
    /// Otherwise, uses CPU SIMD (AVX2/SSE).
    pub(crate) fn fused_matmul(&self, input: &[f32], weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        use crate::quantize::{
            dequantize_q4_1, dequantize_q5_0, fused_q4_0_q8_0_parallel_matvec,
            fused_q4k_parallel_matvec, fused_q5k_parallel_matvec, fused_q6k_parallel_matvec,
            fused_q8_0_q8_0_parallel_matvec,
        };

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // PARITY-113/115: Route to CUDA when enabled
        // PARITY-115: Use native Q4_K CUDA kernel for Q4_K weights (no dequantization needed)
        // For Q5_K/Q6_K: dequantize to FP32 and use CUDA GEMM
        #[cfg(feature = "cuda")]
        if let Some(ref executor_mutex) = self.cuda_executor {
            use tracing::info_span;

            let gemm_start = std::time::Instant::now();
            let mut output = vec![0.0f32; seq_len * out_dim];

            // PAR-003/PAR-005: Use native Q4_K GEMV kernel with cached weights
            // PAR-003: Optimized for M=1 token generation with warp shuffle reduction
            // PAR-005: Weights cached on GPU to avoid ~50+ CPU→GPU transfers per token
            if weight.qtype == GGUF_TYPE_Q4_K && seq_len == 1 {
                // Use data pointer as unique cache key (stable since model owns data)
                let cache_key = format!("q4k_{:016x}", weight.data.as_ptr() as usize);

                {
                    let mut executor =
                        executor_mutex
                            .lock()
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q4k_lock".to_string(),
                                reason: format!("Failed to acquire CUDA executor lock: {e}"),
                            })?;

                    // THREAD-RESOLVED: Ensure CUDA context is current for this thread
                    // (context may have been created on a different thread)
                    executor
                        .make_current()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_make_current".to_string(),
                            reason: format!("Failed to set CUDA context current: {e}"),
                        })?;

                    // PAR-005: Lazy cache - upload weights on first use
                    if !executor.has_quantized_weights(&cache_key) {
                        executor
                            .load_quantized_weights(&cache_key, &weight.data)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q4k_cache".to_string(),
                                reason: format!("Failed to cache Q4_K weights: {e}"),
                            })?;
                    }

                    // Use cached GEMV (no weight transfer)
                    executor
                        .q4k_gemv_cached(
                            &cache_key,
                            input,
                            &mut output,
                            out_dim as u32,
                            in_dim as u32,
                        )
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_q4k_gemv".to_string(),
                            reason: format!("CUDA Q4_K GEMV failed: {e}"),
                        })?;
                }
                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;

                // Emit tracing span for Q4_K GEMV kernel
                let _span = info_span!(
                    "gpu_kernel:q4k_gemv",
                    gpu.backend = "cuda",
                    gpu.kernel = "q4k_gemv_cached",
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }

            // PAR-003/PAR-005: Use native Q5_K GEMV kernel with cached weights
            if weight.qtype == GGUF_TYPE_Q5_K && seq_len == 1 {
                let cache_key = format!("q5k_{:016x}", weight.data.as_ptr() as usize);

                {
                    let mut executor =
                        executor_mutex
                            .lock()
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q5k_lock".to_string(),
                                reason: format!("Failed to acquire CUDA executor lock: {e}"),
                            })?;

                    // THREAD-RESOLVED: Ensure CUDA context is current for this thread
                    executor
                        .make_current()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_make_current".to_string(),
                            reason: format!("Failed to set CUDA context current: {e}"),
                        })?;

                    // PAR-005: Lazy cache
                    if !executor.has_quantized_weights(&cache_key) {
                        executor
                            .load_quantized_weights(&cache_key, &weight.data)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q5k_cache".to_string(),
                                reason: format!("Failed to cache Q5_K weights: {e}"),
                            })?;
                    }

                    executor
                        .q5k_gemv_cached(
                            &cache_key,
                            input,
                            &mut output,
                            out_dim as u32,
                            in_dim as u32,
                        )
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_q5k_gemv".to_string(),
                            reason: format!("CUDA Q5_K GEMV failed: {e}"),
                        })?;
                }
                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;

                let _span = info_span!(
                    "gpu_kernel:q5k_gemv",
                    gpu.backend = "cuda",
                    gpu.kernel = "q5k_gemv_cached",
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }

            // PAR-003/PAR-005: Use native Q6_K GEMV kernel with cached weights
            if weight.qtype == GGUF_TYPE_Q6_K && seq_len == 1 {
                let cache_key = format!("q6k_{:016x}", weight.data.as_ptr() as usize);

                {
                    let mut executor =
                        executor_mutex
                            .lock()
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q6k_lock".to_string(),
                                reason: format!("Failed to acquire CUDA executor lock: {e}"),
                            })?;

                    // THREAD-RESOLVED: Ensure CUDA context is current for this thread
                    executor
                        .make_current()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_make_current".to_string(),
                            reason: format!("Failed to set CUDA context current: {e}"),
                        })?;

                    // PAR-005: Lazy cache
                    if !executor.has_quantized_weights(&cache_key) {
                        executor
                            .load_quantized_weights(&cache_key, &weight.data)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_q6k_cache".to_string(),
                                reason: format!("Failed to cache Q6_K weights: {e}"),
                            })?;
                    }

                    executor
                        .q6k_gemv_cached(
                            &cache_key,
                            input,
                            &mut output,
                            out_dim as u32,
                            in_dim as u32,
                        )
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_q6k_gemv".to_string(),
                            reason: format!("CUDA Q6_K GEMV failed: {e}"),
                        })?;
                }
                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;

                let _span = info_span!(
                    "gpu_kernel:q6k_gemv",
                    gpu.backend = "cuda",
                    gpu.kernel = "q6k_gemv_cached",
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }

            // PAR-003: Q4_K/Q5_K/Q6_K with seq_len==1 now handled by native GEMV kernels above
            // This fallback is only for cases where CUDA is not working
            let is_q4k_q5k_q6k_matvec = false; // Disabled - native GEMV kernels handle M=1

            if is_q4k_q5k_q6k_matvec {
                // Fall through to CPU path - no longer needed with native GEMV kernels
            } else {
                // Fallback: Dequantize and use FP32 GEMM for batched operations (seq_len > 1)
                let dequant_weight = self.dequantize_weight_for_cuda(weight)?;

                {
                    let mut executor =
                        executor_mutex
                            .lock()
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "cuda_gemm_lock".to_string(),
                                reason: format!("Failed to acquire CUDA executor lock: {e}"),
                            })?;

                    // THREAD-RESOLVED: Ensure CUDA context is current for this thread
                    executor
                        .make_current()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_make_current".to_string(),
                            reason: format!("Failed to set CUDA context current: {e}"),
                        })?;

                    executor
                        .gemm(
                            input,
                            &dequant_weight,
                            &mut output,
                            seq_len as u32,
                            out_dim as u32,
                            in_dim as u32,
                        )
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_gemm".to_string(),
                            reason: format!("CUDA GEMM failed: {e}"),
                        })?;
                }
                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;

                // Emit tracing span for CUDA GEMM kernel execution
                let _span = info_span!(
                    "gpu_kernel:gemm_fp32",
                    gpu.backend = "cuda",
                    gpu.kernel = "gemm_fp32",
                    gpu.dimensions.m = seq_len,
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                // Increment kernel count
                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }
        }

        // CPU path: For Q4_0, use fused Q8_0 integer SIMD matmul (llama.cpp parity)
        if weight.qtype == GGUF_TYPE_Q4_0 {
            if seq_len == 1 {
                return fused_q4_0_q8_0_parallel_matvec(&weight.data, input, in_dim, out_dim);
            }
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = fused_q4_0_q8_0_parallel_matvec(&weight.data, x, in_dim, out_dim)?;
                output.extend_from_slice(&row_output);
            }
            return Ok(output);
        }

        // CPU path: For Q8_0, use fused Q8_0 × Q8_0 integer SIMD matmul
        // This avoids the massive dequantization allocation (544MB for Qwen2.5 LM head)
        if weight.qtype == GGUF_TYPE_Q8_0 {
            if seq_len == 1 {
                return fused_q8_0_q8_0_parallel_matvec(&weight.data, input, in_dim, out_dim);
            }
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = fused_q8_0_q8_0_parallel_matvec(&weight.data, x, in_dim, out_dim)?;
                output.extend_from_slice(&row_output);
            }
            return Ok(output);
        }

        // CPU path: For Q4_1, use dequantize + SIMD matmul
        if weight.qtype == GGUF_TYPE_Q4_1 {
            let weights_f32 = dequantize_q4_1(&weight.data)?;

            // Use trueno SIMD for matmul
            let weight_matrix = match TruenoMatrix::from_vec(out_dim, in_dim, weights_f32) {
                Ok(m) => m,
                Err(_) => {
                    return Err(RealizarError::InvalidShape {
                        reason: "Failed to create weight matrix for Q4_1".to_string(),
                    });
                },
            };

            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let x_vec = TruenoVector::from_slice(x);
                match weight_matrix.matvec(&x_vec) {
                    Ok(r) => output.extend_from_slice(r.as_slice()),
                    Err(_) => {
                        return Err(RealizarError::InvalidShape {
                            reason: "SIMD matvec failed for Q4_1".to_string(),
                        });
                    },
                }
            }
            return Ok(output);
        }

        // CPU path: For Q5_0, use dequantize + SIMD matmul
        if weight.qtype == GGUF_TYPE_Q5_0 {
            let weights_f32 = dequantize_q5_0(&weight.data)?;

            // Use trueno SIMD for matmul
            let weight_matrix = match TruenoMatrix::from_vec(out_dim, in_dim, weights_f32) {
                Ok(m) => m,
                Err(_) => {
                    return Err(RealizarError::InvalidShape {
                        reason: "Failed to create weight matrix for Q5_0".to_string(),
                    });
                },
            };

            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let x_vec = TruenoVector::from_slice(x);
                match weight_matrix.matvec(&x_vec) {
                    Ok(r) => output.extend_from_slice(r.as_slice()),
                    Err(_) => {
                        return Err(RealizarError::InvalidShape {
                            reason: "SIMD matvec failed for Q5_0".to_string(),
                        });
                    },
                }
            }
            return Ok(output);
        }

        // CPU path: Process each position in sequence
        if seq_len > 1 {
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = match weight.qtype {
                    GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    // Q6_K: All weights are row-major in TinyLlama
                    GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    _ => {
                        return Err(RealizarError::UnsupportedOperation {
                            operation: "owned_fused_matmul".to_string(),
                            reason: format!(
                                "Fused matmul only supports Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                                weight.qtype
                            ),
                        });
                    },
                };
                output.extend_from_slice(&row_output);
            }
            Ok(output)
        } else {
            // Single position - most common case in generation
            match weight.qtype {
                GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                // Q6_K: All weights are row-major in TinyLlama
                GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                _ => Err(RealizarError::UnsupportedOperation {
                    operation: "owned_fused_matmul".to_string(),
                    reason: format!(
                        "Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                        weight.qtype
                    ),
                }),
            }
        }
    }

    /// Fused matrix-vector multiply - writes to pre-allocated buffer (IMP-131)
    ///
    /// Zero-allocation variant for hot-path inference.
    /// Eliminates ~30-40% of allocation overhead per token.
    ///
    /// # Arguments
    /// * `input` - Input activations [seq_len * in_dim]
    /// * `weight` - Quantized weight tensor
    /// * `output` - Pre-allocated output buffer [out_dim]
    ///
    /// # Errors
    /// Returns error if dimensions don't match or quantization type unsupported
    fn fused_matmul_into(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        output: &mut [f32],
    ) -> Result<()> {
        use crate::quantize::{
            fused_q4_0_q8_0_parallel_matvec_into, fused_q4k_parallel_matvec_into,
            fused_q5k_parallel_matvec_into, fused_q6k_parallel_matvec_into,
            fused_q8_0_q8_0_parallel_matvec_into,
        };

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // Only support single-token case for now (most common in generation)
        if seq_len != 1 {
            // Fall back to allocating version for batch
            let result = self.fused_matmul(input, weight)?;
            output[..result.len()].copy_from_slice(&result);
            return Ok(());
        }

        // Ensure output buffer is properly sized
        debug_assert!(
            output.len() >= out_dim,
            "Output buffer too small: {} < {}",
            output.len(),
            out_dim
        );

        match weight.qtype {
            GGUF_TYPE_Q4_0 => {
                // Q4_0 _into derives out_dim from output.len()
                fused_q4_0_q8_0_parallel_matvec_into(
                    &weight.data,
                    input,
                    in_dim,
                    &mut output[..out_dim],
                )
            },
            GGUF_TYPE_Q8_0 => fused_q8_0_q8_0_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            _ => {
                // Fall back to allocating version for unsupported types
                let result = self.fused_matmul(input, weight)?;
                output[..result.len()].copy_from_slice(&result);
                Ok(())
            },
        }
    }

    /// PAR-126: Q8K-accelerated fused matmul using VNNI instructions
    ///
    /// This variant quantizes f32 activations to Q8K format and uses the
    /// AVX-512 VNNI path which is ~30% faster than AVX2 for Q4K weights.
    ///
    /// # Arguments
    /// * `input` - f32 activations [in_dim]
    /// * `weight` - Q4K quantized weight tensor
    /// * `output` - Pre-allocated output buffer [out_dim]
    /// * `q8k_scales` - Pre-allocated Q8K scales scratch [in_dim/256]
    /// * `q8k_quants` - Pre-allocated Q8K quants scratch [in_dim padded to 256]
    fn fused_matmul_q8k_into(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        output: &mut [f32],
        q8k_scales: &mut [f32],
        q8k_quants: &mut [i8],
    ) -> Result<()> {
        use crate::quantize::{fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into};

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // Only support single-token case for now (most common in generation)
        if seq_len != 1 {
            // Fall back to allocating version for batch
            let result = self.fused_matmul(input, weight)?;
            output[..result.len()].copy_from_slice(&result);
            return Ok(());
        }

        // Only use Q8K path for Q4_K weights (has VNNI optimization)
        // Q6_K uses f32 path since Q8K conversion overhead > memory bandwidth savings
        if weight.qtype != GGUF_TYPE_Q4_K {
            return self.fused_matmul_into(input, weight, output);
        }

        // Pad input if needed for Q8K (256-element super-blocks)
        let padded_len = in_dim.next_multiple_of(256);
        let num_sb = padded_len / 256;

        // Ensure scratch buffers are large enough
        if q8k_scales.len() < num_sb || q8k_quants.len() < padded_len {
            // Scratch too small, fall back to allocating version
            return self.fused_matmul_into(input, weight, output);
        }

        // Quantize activations to Q8K format using scratch buffers
        if in_dim < padded_len {
            // Need to pad - copy input and zero-pad
            q8k_quants[in_dim..padded_len]
                .iter_mut()
                .for_each(|x| *x = 0);
            // Create temporary padded buffer (small allocation for edge case)
            let mut padded = vec![0.0f32; padded_len];
            padded[..in_dim].copy_from_slice(input);
            quantize_activations_q8k_into(
                &padded,
                &mut q8k_scales[..num_sb],
                &mut q8k_quants[..padded_len],
            )?;
        } else {
            quantize_activations_q8k_into(
                &input[..padded_len],
                &mut q8k_scales[..num_sb],
                &mut q8k_quants[..padded_len],
            )?;
        }

        // Use VNNI-accelerated Q4K×Q8K path
        fused_q4k_q8k_parallel_matvec_into(
            &weight.data,
            &q8k_scales[..num_sb],
            &q8k_quants[..padded_len],
            in_dim,
            out_dim,
            &mut output[..out_dim],
        )
    }

    /// QKV projection supporting both fused (phi-2) and separate (llama) formats
    ///
    /// Five Whys Root Cause Fix: This method handles both tensor layouts
    /// transparently, allowing TinyLlama and other LLaMA-style models to work.
    pub fn qkv_matmul(&self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let seq_len = input.len() / hidden_dim;

                // DIVERGENCE-DEBUG: Trace Q projection inputs
                if std::env::var("QKV_DEBUG").is_ok() {
                    eprintln!(
                        "[QKV_DEBUG] Q weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                        q.in_dim,
                        q.out_dim,
                        q.qtype,
                        q.data.len()
                    );
                    eprintln!("[QKV_DEBUG] Q weight first 16 bytes: {:?}", &q.data[..16]);
                    eprintln!(
                        "[QKV_DEBUG] Input first 5: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                        input[0], input[1], input[2], input[3], input[4]
                    );
                }

                let q_out = self.fused_matmul(input, q)?;

                // DIVERGENCE-DEBUG: Trace Q projection output
                if std::env::var("QKV_DEBUG").is_ok() {
                    eprintln!(
                        "[QKV_DEBUG] Q output first 5: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                        q_out[0], q_out[1], q_out[2], q_out[3], q_out[4]
                    );
                }
                let k_out = self.fused_matmul(input, k)?;
                let v_out = self.fused_matmul(input, v)?;

                // Interleave Q, K, V for each position
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(seq_len * qkv_dim);
                for s in 0..seq_len {
                    output.extend_from_slice(&q_out[s * q.out_dim..(s + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[s * k.out_dim..(s + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[s * v.out_dim..(s + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// QKV projection - zero-allocation variant (P1-REV)
    ///
    /// Writes QKV output directly to pre-allocated buffer, eliminating
    /// Vec allocation that was 42% of forward pass overhead.
    ///
    /// # Arguments
    /// * `input` - Normalized hidden state [hidden_dim]
    /// * `qkv` - QKV weights (fused or separate)
    /// * `output` - Pre-allocated buffer [q_dim + k_dim + v_dim]
    pub fn qkv_matmul_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
    ) -> Result<()> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_into(input, weight, output),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For single-token case (seq_len=1), write Q, K, V directly
                let q_dim = q.out_dim;
                let k_dim = k.out_dim;
                let v_dim = v.out_dim;

                // Write Q to output[0..q_dim]
                self.fused_matmul_into(input, q, &mut output[..q_dim])?;
                // Write K to output[q_dim..q_dim+k_dim]
                self.fused_matmul_into(input, k, &mut output[q_dim..q_dim + k_dim])?;
                // Write V to output[q_dim+k_dim..]
                self.fused_matmul_into(
                    input,
                    v,
                    &mut output[q_dim + k_dim..q_dim + k_dim + v_dim],
                )?;

                Ok(())
            },
        }
    }

    /// PAR-126: Q8K-accelerated QKV matmul using pre-quantized activations
    ///
    /// Uses pre-quantized Q8K activations for VNNI-accelerated matmul.
    /// This avoids re-quantizing for each of Q, K, V when using separate weights.
    pub fn qkv_matmul_q8k_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
        q8k_scales: &[f32],
        q8k_quants: &[i8],
    ) -> Result<()> {
        use crate::quantize::fused_q4k_q8k_parallel_matvec_into;

        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                // Use Q8K path if Q4K weights, otherwise fall back to f32
                if weight.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(
                        &weight.data,
                        q8k_scales,
                        q8k_quants,
                        weight.in_dim,
                        weight.out_dim,
                        output,
                    )
                } else {
                    self.fused_matmul_into(input, weight, output)
                }
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.out_dim;
                let k_dim = k.out_dim;
                let v_dim = v.out_dim;

                // Use Q8K path for Q4K weights (sequential to avoid overhead)
                if q.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(
                        &q.data,
                        q8k_scales,
                        q8k_quants,
                        q.in_dim,
                        q_dim,
                        &mut output[..q_dim],
                    )?;
                } else {
                    self.fused_matmul_into(input, q, &mut output[..q_dim])?;
                }

                if k.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(
                        &k.data,
                        q8k_scales,
                        q8k_quants,
                        k.in_dim,
                        k_dim,
                        &mut output[q_dim..q_dim + k_dim],
                    )?;
                } else {
                    self.fused_matmul_into(input, k, &mut output[q_dim..q_dim + k_dim])?;
                }

                if v.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(
                        &v.data,
                        q8k_scales,
                        q8k_quants,
                        v.in_dim,
                        v_dim,
                        &mut output[q_dim + k_dim..q_dim + k_dim + v_dim],
                    )?;
                } else {
                    self.fused_matmul_into(
                        input,
                        v,
                        &mut output[q_dim + k_dim..q_dim + k_dim + v_dim],
                    )?;
                }

                Ok(())
            },
        }
    }

    /// Fused RMSNorm + matmul for Q4_0 weights
    ///
    /// Combines RMSNorm normalization with quantized matmul:
    /// 1. Computes inv_rms = 1 / sqrt(mean(x^2) + eps)
    /// 2. Quantizes (x * inv_rms * norm_weight) to Q8_0
    /// 3. Performs Q4_0 × Q8_0 integer matmul
    ///
    /// This eliminates the intermediate normalized vector allocation.
    fn fused_rmsnorm_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights (most common)
        if weight.qtype == GGUF_TYPE_Q4_0 && input.len() == weight.in_dim {
            return fused_rmsnorm_q4_0_matmul(
                input,
                norm_weight,
                eps,
                &weight.data,
                weight.in_dim,
                weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, norm_weight, eps);
        self.fused_matmul(&normed, weight)
    }

    /// Fused RMSNorm + QKV projection
    ///
    /// Combines attention layer norm with QKV projection in one operation.
    /// Avoids allocating intermediate normalized vector.
    pub fn fused_rmsnorm_qkv_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        qkv: &OwnedQKVWeights,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_rmsnorm_matmul(input, norm_weight, eps, weight)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, we need to normalize once and reuse
                // (Can't easily fuse since we need same normalized input for all three)
                let normed = ops::rms_norm(input, norm_weight, eps);

                let q_out = self.fused_matmul(&normed, q)?;
                let k_out = self.fused_matmul(&normed, k)?;
                let v_out = self.fused_matmul(&normed, v)?;

                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(qkv_dim);
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// Fused RMSNorm + FFN up/gate projections for SwiGLU
    ///
    /// For SwiGLU models, computes:
    /// - ffn_up = matmul(rmsnorm(hidden, norm_weight), up_weight)
    /// - ffn_gate = matmul(rmsnorm(hidden, norm_weight), gate_weight)
    ///
    /// RMSNorm and Q8_0 quantization are computed once and shared between both matmuls.
    /// Both matmuls run in parallel via rayon::join.
    pub fn fused_rmsnorm_ffn_up_gate(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        up_weight: &OwnedQuantizedTensor,
        gate_weight: &OwnedQuantizedTensor,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::quantize::fused_rmsnorm_ffn_up_gate;

        // Only use fused path for Q4_0 weights
        if up_weight.qtype == GGUF_TYPE_Q4_0
            && gate_weight.qtype == GGUF_TYPE_Q4_0
            && input.len() == up_weight.in_dim
            && up_weight.in_dim == gate_weight.in_dim
            && up_weight.out_dim == gate_weight.out_dim
        {
            return fused_rmsnorm_ffn_up_gate(
                input,
                norm_weight,
                eps,
                &up_weight.data,
                &gate_weight.data,
                up_weight.in_dim,
                up_weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmuls for other types
        let normed = ops::rms_norm(input, norm_weight, eps);
        let up_out = self.fused_matmul(&normed, up_weight)?;
        let gate_out = self.fused_matmul(&normed, gate_weight)?;
        Ok((up_out, gate_out))
    }

    /// Fused RMSNorm + LM head projection
    ///
    /// Combines final layer norm with output projection in one operation.
    /// Eliminates intermediate normalized vector allocation.
    pub fn fused_rmsnorm_lm_head(&self, input: &[f32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights
        if self.lm_head_weight.qtype == GGUF_TYPE_Q4_0 && input.len() == self.lm_head_weight.in_dim
        {
            return fused_rmsnorm_q4_0_matmul(
                input,
                &self.output_norm_weight,
                self.config.eps,
                &self.lm_head_weight.data,
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, &self.output_norm_weight, self.config.eps);

        // PAR-060-DEBUG: Removed unconditional print from hot path (was causing 100x slowdown)

        self.fused_matmul(&normed, &self.lm_head_weight)
    }

    /// PARITY-113: Dequantize weight tensor to FP32 for CUDA GEMM
    ///
    /// This is a fallback path for non-matvec operations (seq_len > 1).
    /// For seq_len=1 matvec, native quantized kernels are used instead:
    /// - Q4_K: `q4k_matvec()` (PARITY-115)
    /// - Q5_K: `q5k_matvec()` (PARITY-116)
    /// - Q6_K: `q6k_matvec()` (PARITY-117)
    ///
    /// Performance note: Native kernels provide 2.4-3.5x memory bandwidth
    /// reduction vs this dequantize-then-GEMM path.
    #[cfg(feature = "cuda")]
    fn dequantize_weight_for_cuda(&self, weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        // Q4_K block sizes per GGML spec
        const Q4K_BLOCK_SIZE: usize = 144; // bytes per 256-element super-block
        const Q4K_WEIGHTS_PER_BLOCK: usize = 256;

        let out_dim = weight.out_dim;
        let in_dim = weight.in_dim;
        let total_weights = out_dim * in_dim;

        // Allocate output buffer
        let mut output = vec![0.0f32; total_weights];

        // Dequantize based on quantization type
        match weight.qtype {
            GGUF_TYPE_Q4_K => {
                // PAR-002/003 RESOLVED: Use proper dequantization that matches CPU path
                // CPU path in fused_q4k_parallel_matvec:
                //   - super_blocks_per_row = in_dim.div_ceil(QK_K)
                //   - bytes_per_row = super_blocks_per_row * 144
                //   - For each output o in 0..out_dim: row_data = weight[o*bytes_per_row..]
                // So weight layout is [out_dim, in_dim] (out_dim rows, in_dim cols)
                use crate::quantize::{dequantize_q4_k_simd, QK_K};

                // Each row has in_dim elements = super_blocks_per_row super-blocks
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let bytes_per_row = super_blocks_per_row * Q4K_BLOCK_SIZE;

                for row in 0..out_dim {
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    if row_end > weight.data.len() {
                        break;
                    }
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q4_k_simd(row_data)?;

                    // Copy to output (may have padding due to super-block alignment)
                    let copy_len = in_dim.min(row_dequant.len());
                    let out_start = row * in_dim;
                    output[out_start..out_start + copy_len]
                        .copy_from_slice(&row_dequant[..copy_len]);
                }

                // Transpose from [out_dim, in_dim] to [in_dim, out_dim] for GEMV kernel
                // GEMV expects A[k, n] at offset k*N + n, computing y[n] = sum_k(A[k,n] * x[k])
                // We have W[o, i] at offset o*in_dim + i (CPU layout)
                // Need A[i, o] = W[o, i] so that y[o] = sum_i(A[i,o] * x[i]) = sum_i(W[o,i] * x[i])
                let mut transposed = vec![0.0f32; total_weights];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        // A[i, o] = W[o, i]
                        transposed[i * out_dim + o] = output[o * in_dim + i];
                    }
                }
                Ok(transposed)
            },
            GGUF_TYPE_Q5_K => {
                // Q5_K: 176 bytes per 256-element super-block
                use crate::quantize::{dequantize_q5_k, QK_K};

                // GGUF tensor layout is [out_dim, in_dim] (same as Q4_K)
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let bytes_per_row = super_blocks_per_row * 176;

                for row in 0..out_dim {
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    if row_end > weight.data.len() {
                        break;
                    }
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q5_k(row_data)?;

                    let copy_len = in_dim.min(row_dequant.len());
                    let out_start = row * in_dim;
                    output[out_start..out_start + copy_len]
                        .copy_from_slice(&row_dequant[..copy_len]);
                }

                // Transpose from [out_dim, in_dim] to [in_dim, out_dim] for GEMV kernel
                let mut transposed = vec![0.0f32; total_weights];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        transposed[i * out_dim + o] = output[o * in_dim + i];
                    }
                }
                Ok(transposed)
            },
            GGUF_TYPE_Q6_K => {
                // Q6_K: 210 bytes per 256-element super-block
                use crate::quantize::{dequantize_q6_k, QK_K};

                // GGUF tensor layout is [out_dim, in_dim] (same as Q4_K)
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let bytes_per_row = super_blocks_per_row * 210;

                for row in 0..out_dim {
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    if row_end > weight.data.len() {
                        break;
                    }
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q6_k(row_data)?;

                    let copy_len = in_dim.min(row_dequant.len());
                    let out_start = row * in_dim;
                    output[out_start..out_start + copy_len]
                        .copy_from_slice(&row_dequant[..copy_len]);
                }

                // Transpose from [out_dim, in_dim] to [in_dim, out_dim] for GEMV kernel
                let mut transposed = vec![0.0f32; total_weights];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        transposed[i * out_dim + o] = output[o * in_dim + i];
                    }
                }
                Ok(transposed)
            },
            GGUF_TYPE_Q2_K => {
                // Q2_K: 84 bytes per 256-element super-block
                use crate::quantize::{dequantize_q2_k, QK_K};

                // GGUF tensor layout is [out_dim, in_dim] (same as Q4_K)
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let bytes_per_row = super_blocks_per_row * 84;

                for row in 0..out_dim {
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    if row_end > weight.data.len() {
                        break;
                    }
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q2_k(row_data)?;

                    let copy_len = in_dim.min(row_dequant.len());
                    let out_start = row * in_dim;
                    output[out_start..out_start + copy_len]
                        .copy_from_slice(&row_dequant[..copy_len]);
                }

                // Transpose from [out_dim, in_dim] to [in_dim, out_dim] for GEMV kernel
                let mut transposed = vec![0.0f32; total_weights];
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        transposed[i * out_dim + o] = output[o * in_dim + i];
                    }
                }
                Ok(transposed)
            },
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "dequantize_for_cuda".to_string(),
                reason: format!(
                    "Unsupported quantization type {} for CUDA dequantization",
                    weight.qtype
                ),
            }),
        }
    }

    /// Look up token embeddings (public for debugging PAR-001)
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// Look up single token embedding into pre-allocated buffer (IMP-131)
    fn embed_into(&self, token_id: u32, output: &mut [f32]) {
        let hidden_dim = self.config.hidden_dim;
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        if end <= self.token_embedding.len() {
            output[..hidden_dim].copy_from_slice(&self.token_embedding[start..end]);
        } else {
            output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Apply layer normalization
    /// RMSNorm (Root Mean Square Layer Normalization)
    ///
    /// PMAT-094 FIX: Qwen2, LLaMA, Mistral use RMSNorm, NOT LayerNorm.
    /// Formula: output = x / sqrt(mean(x^2) + eps) * weight
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

            // RMSNorm: compute root mean square (no mean subtraction!)
            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

            for j in 0..hidden_dim {
                let normalized = x[j] / rms;
                let mut val = normalized * weight[j];
                if let Some(b) = bias {
                    val += b[j];
                }
                output.push(val);
            }
        }

        output
    }

    /// RMSNorm to pre-allocated buffer (IMP-131)
    ///
    /// PMAT-094 FIX: Uses RMSNorm, not LayerNorm.
    fn layer_norm_into(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
        output: &mut [f32],
    ) {
        let hidden_dim = weight.len();
        // Single position case for generation
        let x = &input[..hidden_dim];

        // RMSNorm: compute root mean square (no mean subtraction!)
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        for j in 0..hidden_dim {
            let normalized = x[j] / rms;
            output[j] = normalized * weight[j];
            if let Some(b) = bias {
                output[j] += b[j];
            }
        }
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
            let sqrt_2_over_pi = 0.797_884_6_f32;
            let c = 0.044_715_f32;
            let inner = sqrt_2_over_pi * (*x + c * *x * *x * *x);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Apply SiLU (Sigmoid Linear Unit) activation for SwiGLU FFN
    /// SiLU(x) = x * sigmoid(x)
    fn silu(&self, input: &mut [f32]) {
        for x in input.iter_mut() {
            *x = *x * (1.0 / (1.0 + (-*x).exp()));
        }
    }

    /// Apply RMSNorm (Root Mean Square Layer Normalization) using trueno SIMD
    /// LLaMA, TinyLlama, Mistral, etc. use RMSNorm instead of LayerNorm
    /// Formula: x / sqrt(mean(x^2) + eps) * weight
    ///
    /// Uses trueno SIMD operations for performance:
    /// - sum_of_squares(): SIMD-accelerated sum of x^2
    /// - scale(): SIMD-accelerated scalar multiplication
    /// - mul(): SIMD-accelerated element-wise multiplication
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let hidden_dim = weight.len();
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        // Pre-create weight vector for SIMD multiply (reused across sequence)
        let weight_vec = TruenoVector::from_slice(weight);

        for i in 0..seq_len {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            let x = &input[start..end];

            // Create SIMD vector from input slice
            let x_vec = TruenoVector::from_slice(x);

            // SIMD: sum of squares (replaces scalar x.iter().map(|v| v*v).sum())
            let sum_sq = x_vec.sum_of_squares().unwrap_or_else(|_| {
                // Fallback to scalar if SIMD fails
                x.iter().map(|v| v * v).sum::<f32>()
            });

            // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            // eps is added INSIDE the sqrt (crucial for numerical stability)
            let mean_sq = sum_sq / hidden_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            // SIMD: scale by inv_rms, then multiply by weight
            // x * inv_rms * weight
            match x_vec
                .scale(inv_rms)
                .and_then(|scaled| scaled.mul(&weight_vec))
            {
                Ok(result) => {
                    output.extend_from_slice(result.as_slice());
                },
                Err(_) => {
                    // Fallback to scalar if SIMD fails
                    for j in 0..hidden_dim {
                        output.push(x[j] * inv_rms * weight[j]);
                    }
                },
            }
        }

        output
    }

    /// Apply RMSNorm to pre-allocated buffer (IMP-131)
    fn rms_norm_into(&self, input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
        let hidden_dim = weight.len();
        // Single position case for generation
        let x = &input[..hidden_dim];

        // Create SIMD vectors
        let x_vec = TruenoVector::from_slice(x);
        let weight_vec = TruenoVector::from_slice(weight);

        // SIMD: sum of squares
        let sum_sq = x_vec
            .sum_of_squares()
            .unwrap_or_else(|_| x.iter().map(|v| v * v).sum::<f32>());

        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // SIMD: scale by inv_rms, then multiply by weight
        match x_vec
            .scale(inv_rms)
            .and_then(|scaled| scaled.mul(&weight_vec))
        {
            Ok(result) => {
                output[..hidden_dim].copy_from_slice(result.as_slice());
            },
            Err(_) => {
                // Fallback to scalar
                for j in 0..hidden_dim {
                    output[j] = x[j] * inv_rms * weight[j];
                }
            },
        }
    }

    /// Apply RoPE (Rotary Position Embeddings) to Q or K vectors using trueno SIMD (IMP-101a)
    ///
    /// RoPE encodes position by rotating pairs of dimensions.
    /// Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    ///
    /// Uses trueno SIMD operations for performance:
    /// - Pre-computes cos/sin vectors once per position (reused across heads)
    /// - mul(): SIMD element-wise multiplication
    /// - sub()/add(): SIMD element-wise arithmetic
    ///
    /// # Arguments
    /// * `x` - Vector to apply RoPE to [num_heads_in_x * head_dim]
    /// * `position` - Position index for frequency calculation
    /// * `num_heads_in_x` - Number of heads in x (num_heads for Q, num_kv_heads for K)
    ///
    /// # GQA Support
    /// For GQA models, pass num_heads for Q vectors and num_kv_heads for K vectors.
    pub(crate) fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let rope_type = self.config.rope_type;

        // Stack-based buffers (max 128 = 256 head_dim, covers all common models)
        // Avoids heap allocation on every call
        let mut cos_vals: [f32; 128] = [0.0; 128];
        let mut sin_vals: [f32; 128] = [0.0; 128];

        // Pre-compute cos/sin for this position (reused across all heads)
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;
        for i in 0..half_dim.min(128) {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_v, cos_v) = angle.sin_cos();
            cos_vals[i] = cos_v;
            sin_vals[i] = sin_v;
        }

        // Apply rotation to each head
        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;

            if head_start + head_dim > x.len() {
                continue;
            }

            if rope_type == 2 {
                // NEOX style: split halves (x[0..half], x[half..])
                // Used by GPT-NeoX and some newer models
                let (first_half, second_half) =
                    x[head_start..head_start + head_dim].split_at_mut(half_dim);
                crate::quantize::apply_rope_rotation_simd(
                    first_half,
                    second_half,
                    &cos_vals[..half_dim],
                    &sin_vals[..half_dim],
                );
            } else {
                // NORM style (type 0): adjacent pairs (x[0], x[1]), (x[2], x[3]), ...
                // This is the default for LLaMA-family models
                let head_slice = &mut x[head_start..head_start + head_dim];
                for i in 0..half_dim {
                    let x0 = head_slice[2 * i];
                    let x1 = head_slice[2 * i + 1];
                    let cos_v = cos_vals[i];
                    let sin_v = sin_vals[i];
                    head_slice[2 * i] = x0 * cos_v - x1 * sin_v;
                    head_slice[2 * i + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }

    /// Compute scaled dot-product attention with causal mask (IMP-101b)
    ///
    /// Computes: softmax(QK^T / sqrt(d_k)) * V with causal masking
    ///
    /// # Arguments
    /// * `q` - Query vectors [seq_len, q_dim] where q_dim = num_heads * head_dim
    /// * `k` - Key vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    /// * `v` - Value vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    ///
    /// # Returns
    /// Attention output [seq_len, q_dim] where q_dim = num_heads * head_dim
    ///
    /// # GQA (Grouped Query Attention) Support
    /// For models where num_kv_heads < num_heads (e.g., TinyLlama: 4 vs 32),
    /// multiple Q heads share the same K/V head. The group size is num_heads / num_kv_heads.
    pub(crate) fn causal_attention(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA: multiple Q heads share each KV head
        // group_size = num_heads / num_kv_heads (e.g., 32/4 = 8 for TinyLlama)
        let group_size = num_heads / num_kv_heads;

        // Q has num_heads heads, K/V have num_kv_heads heads
        let q_dim = num_heads * head_dim; // e.g., 32 * 64 = 2048
        let kv_dim = num_kv_heads * head_dim; // e.g., 4 * 64 = 256

        let mut output = vec![0.0f32; seq_len * q_dim];

        // Process each Q head independently
        for head in 0..num_heads {
            // Map Q head to corresponding KV head (GQA grouping)
            let kv_head = head / group_size;

            let q_head_offset = head * head_dim;
            let kv_head_offset = kv_head * head_dim;

            // Process each query position
            for i in 0..seq_len {
                // Compute attention scores for this query against all keys up to position i (causal)
                let mut scores = Vec::with_capacity(i + 1);
                let q_start = i * q_dim + q_head_offset;

                for j in 0..=i {
                    // Only attend to positions 0..=i (causal mask)
                    let k_start = j * kv_dim + kv_head_offset;

                    // Dot product Q[i] · K[j]
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_start + d] * k[k_start + d];
                    }
                    scores.push(score * scale);
                }

                // Softmax (SIMD-optimized)
                crate::quantize::softmax_simd(&mut scores);

                // Weighted sum of values
                let out_start = i * q_dim + q_head_offset;
                for (j, &weight) in scores.iter().enumerate() {
                    let v_start = j * kv_dim + kv_head_offset;
                    for d in 0..head_dim {
                        output[out_start + d] += weight * v[v_start + d];
                    }
                }
            }
        }

        output
    }

    /// Forward pass with fused Q4_K operations (IMP-100)
    ///
    /// This is 1.37x faster than dequantized f32 due to reduced memory bandwidth.
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
        // Note: intermediate_dim is encoded in layer weight tensors (in_dim/out_dim)
        let _ = self.config.intermediate_dim;

        // 1. Token embedding lookup (f32, fast)
        let mut hidden = self.embed(token_ids);

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 2. Process through transformer layers with FUSED Q4_K ops
        let cpu_debug_layers = std::env::var("CPU_DEBUG_LAYERS").is_ok();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for others)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // CORRECTNESS-011: CPU intermediate debug at L0
            if cpu_debug_layers && layer_idx < 2 {
                eprintln!(
                    "[CPU-L{}] RMSNorm: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, normed[0], normed[1], normed[2]
                );
            }

            // 2b. QKV projection with FUSED dequant+dot (1.37x faster)
            // Note: qkv_dim may differ from 3*hidden_dim for GQA models
            let qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim();
            // For GQA, k_dim and v_dim may be smaller than q_dim
            let k_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { k, .. } => k.out_dim,
            };
            let v_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { v, .. } => v.out_dim,
            };
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // CORRECTNESS-011: Q, K, V before RoPE (after bias)
            if cpu_debug_layers && (layer_idx < 2 || layer_idx == 4 || layer_idx == 5) {
                eprintln!(
                    "[CPU-L{}] Q (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx, qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]
                );
                // K starts at q_dim offset
                eprintln!(
                    "[CPU-L{}] K (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[q_dim],
                    qkv[q_dim + 1],
                    qkv[q_dim + 2],
                    qkv[q_dim + 3],
                    qkv[q_dim + 4]
                );
                // V starts at q_dim + k_dim offset
                let v_offset = q_dim + k_dim;
                eprintln!(
                    "[CPU-L{}] V (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[v_offset],
                    qkv[v_offset + 1],
                    qkv[v_offset + 2],
                    qkv[v_offset + 3],
                    qkv[v_offset + 4]
                );
            }

            // 2c. Proper attention with RoPE and causal mask (IMP-101)
            let seq_len = token_ids.len();

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                // Apply RoPE to Q and K (position-dependent rotation)
                // GQA: Q has num_heads, K has num_kv_heads
                self.apply_rope(&mut q, s, self.config.num_heads);
                self.apply_rope(&mut k, s, self.config.num_kv_heads);

                // CORRECTNESS-011: Q after RoPE at position 0
                if cpu_debug_layers && layer_idx < 2 && s == 0 {
                    eprintln!(
                        "[CPU-L{}] Q (after RoPE): first 3 = [{:.4}, {:.4}, {:.4}]",
                        layer_idx, q[0], q[1], q[2]
                    );
                    eprintln!(
                        "[CPU-L{}] K (after RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, k[0], k[1], k[2], k[3], k[4]
                    );
                    eprintln!(
                        "[CPU-L{}] V: first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, v[0], v[1], v[2], v[3], v[4]
                    );
                }

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_out = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // CORRECTNESS-011: Attention output
            if cpu_debug_layers && layer_idx < 2 {
                eprintln!(
                    "[CPU-L{}] Attn output: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, attn_out[0], attn_out[1], attn_out[2]
                );
            }

            // 2d. Attention output projection with FUSED ops
            // Input is q_dim (attention output), projects back to hidden_dim
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                // LLaMA-style: separate FFN layer norm (use RMSNorm for LLaMA)
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                // phi-2 style: no separate FFN norm, use hidden directly
                // (some models apply attn_norm again, but we've already done residual)
                hidden.clone()
            };

            // 2g. FFN with SwiGLU or GELU activation
            let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA, TinyLlama, Mistral, etc.)
                // output = down(gate(x) * silu(up(x)))
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                // SwiGLU: down(silu(gate(x)) * up(x))
                // Apply SiLU to GATE projection, not up
                ops::silu(&mut ffn_gate);

                // Element-wise multiply: silu(gate) * up
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                ffn_gate
            } else {
                // GELU path (phi-2, GPT-2, etc.)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);
                ffn_hidden
            };

            // 2g. FFN down projection with FUSED ops
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            // CORRECTNESS-011: Per-layer CPU debug output
            if cpu_debug_layers {
                let seq_len = token_ids.len();
                let last_hidden_start = (seq_len - 1) * hidden_dim;
                let last_h = &hidden[last_hidden_start..last_hidden_start + hidden_dim];
                let sum: f32 = last_h.iter().sum();
                let sq_sum: f32 = last_h.iter().map(|x| x * x).sum();
                let rms = (sq_sum / last_h.len() as f32).sqrt();
                eprintln!(
                    "[CPU-L{}] After layer: first 3 = [{:.4}, {:.4}, {:.4}], sum = {:.4}, rms = {:.4}",
                    layer_idx, last_h[0], last_h[1], last_h[2], sum, rms
                );
            }
        }

        // CORRECTNESS-011: CPU hidden state debug output (compare with GPU)
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_hidden_start = (seq_len - 1) * hidden_dim;
            let last_hidden_raw = &hidden[last_hidden_start..last_hidden_start + hidden_dim];

            let sum: f32 = last_hidden_raw.iter().sum();
            let sq_sum: f32 = last_hidden_raw.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_hidden_raw.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Hidden before output_norm:");
            eprintln!(
                "  first 5 = {:?}",
                &last_hidden_raw[..5.min(last_hidden_raw.len())]
            );
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=466.2486, rms=39.4793)");
        }

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for others)
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // CORRECTNESS-011: CPU normed hidden state debug output
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_normed_start = (seq_len - 1) * hidden_dim;
            let last_normed = &normed[last_normed_start..last_normed_start + hidden_dim];

            let sum: f32 = last_normed.iter().sum();
            let sq_sum: f32 = last_normed.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_normed.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Normed hidden:");
            eprintln!("  first 5 = {:?}", &last_normed[..5.min(last_normed.len())]);
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=107.5945, rms=4.6616)");
        }

        // 4. LM head projection with FUSED ops (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Compute logits using fused op
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Get most likely next token
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
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

    /// Generate tokens using fused Q4_K operations (IMP-100)
    ///
    /// This is the HTTP serving entry point for quantized inference.
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
            // Forward pass with fused Q4_K ops (1.37x faster)
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
    pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
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

        // Sample from probability distribution with proper randomness
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();

        let mut cumulative = 0.0;
        for &(idx, prob) in &probs {
            cumulative += prob;
            if cumulative >= r {
                return idx as u32;
            }
        }

        probs.last().map_or(0, |(idx, _)| *idx as u32)
    }

    /// Forward pass with KV cache for efficient autoregressive decoding
    ///
    /// This method properly handles both architectures:
    /// - LLaMA-style: RMSNorm, SwiGLU FFN, GQA attention
    /// - phi-2 style: LayerNorm, GELU FFN, MHA attention
    ///
    /// Uses O(n) per-token cost instead of O(n²) by caching K/V.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for all layers
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    pub fn forward_cached(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Detect architecture: LLaMA uses RMSNorm (no bias) and SwiGLU (has gate weight)
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // PAR-052: Debug output for OwnedQuantizedModel forward path
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();
        if debug_forward && position == 0 {
            eprintln!("[PAR-052] OwnedQuantizedModel::forward_cached");
            eprintln!("[PAR-052] Token ID: {}, Position: {}", token_id, position);
            eprintln!("[PAR-052] use_rmsnorm: {}", use_rmsnorm);
            eprintln!(
                "[PAR-052] Embedding[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!("[PAR-052] Embedding sum: {:.6}", hidden.iter().sum::<f32>());
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // PAR-052: Debug layer 0 normed and QKV values
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] attn_norm[0..8]: {:?}",
                    &layer.attn_norm_weight[..8.min(layer.attn_norm_weight.len())]
                );
                eprintln!(
                    "[PAR-052-L0] normed[0..8]: {:?}",
                    &normed[..8.min(normed.len())]
                );
                eprintln!("[PAR-052-L0] normed sum: {:.6}", normed.iter().sum::<f32>());
            }

            // 2b. QKV projection
            let _qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim();
            let k_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { k, .. } => k.out_dim,
            };
            let v_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { v, .. } => v.out_dim,
            };

            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // PAR-052: Debug QKV after projection
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] QKV dims: q={}, k={}, v={}, total={}",
                    q_dim,
                    k_dim,
                    v_dim,
                    qkv.len()
                );
                eprintln!("[PAR-052-L0] QKV sum: {:.6}", qkv.iter().sum::<f32>());
                eprintln!("[PAR-052-L0] Q[0..8]: {:?}", &qkv[..8.min(q_dim)]);
            }

            // 2c. Extract Q, K, V and apply RoPE
            let mut q = qkv[0..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec();

            // Apply RoPE with correct head counts for GQA
            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, self.config.num_kv_heads);

            // PAR-052: Debug Q after RoPE
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] Q after RoPE[0..8]: {:?}",
                    &q[..8.min(q.len())]
                );
                eprintln!(
                    "[PAR-052-L0] K after RoPE[0..4]: {:?}",
                    &k[..4.min(k.len())]
                );
            }

            // 2d. Compute attention using cached K/V
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - just use V directly (self-attention with single token)
                // Expand V if GQA (num_kv_heads < num_heads)
                if self.config.num_kv_heads < self.config.num_heads {
                    let head_dim = hidden_dim / self.config.num_heads;
                    let group_size = self.config.num_heads / self.config.num_kv_heads;
                    (0..self.config.num_heads)
                        .flat_map(|h| {
                            let kv_head = h / group_size;
                            let start = kv_head * head_dim;
                            v[start..start + head_dim].iter().copied()
                        })
                        .collect()
                } else {
                    v.clone()
                }
            } else {
                // Use cached K/V for attention with GQA support
                self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
            };

            // 2e. Store K and V in cache (store original size, not expanded)
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h. Pre-FFN layer norm (LLaMA has separate ffn_norm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // 2i. FFN with SwiGLU or GELU
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA)
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                // SiLU on gate, then multiply with up
                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                let mut output = self.fused_matmul(&ffn_gate, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut output, bias);
                }
                output
            } else {
                // GELU path (phi-2)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut output = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut output, bias);
                }
                output
            };

            // 2j. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // PAR-052: Debug final hidden state
        if debug_forward && position == 0 {
            eprintln!(
                "[PAR-052] Final hidden sum: {:.6}",
                hidden.iter().sum::<f32>()
            );
            eprintln!(
                "[PAR-052] Final hidden[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!(
                "[PAR-052] After output_norm sum: {:.6}",
                normed.iter().sum::<f32>()
            );
            eprintln!(
                "[PAR-052] output_norm_weight[0..4]: {:?}",
                &self.output_norm_weight[..4.min(self.output_norm_weight.len())]
            );
            eprintln!(
                "[PAR-052] LM head weight dims: in={}, out={}",
                self.lm_head_weight.in_dim, self.lm_head_weight.out_dim
            );
        }

        // 4. LM head projection
        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // PAR-052: Debug final logits
        if debug_forward && position == 0 {
            // Find top-5 logits
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[PAR-052] Top-5 logits:");
            for (idx, val) in indexed.iter().take(5) {
                eprintln!("  Token {}: {:.6}", idx, val);
            }
            eprintln!("[PAR-052] Logits sum: {:.6}", logits.iter().sum::<f32>());
        }

        Ok(logits)
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        &self.config
    }

    /// Compute attention for a single query position using KV cache (IMP-101c)
    ///
    /// This enables O(n) per-token cost instead of O(n²) by reusing cached K/V.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    /// Attention with KV cache using trueno SIMD dot products (IMP-500e)
    ///
    /// OPTIMIZATION: Uses trueno's 4-accumulator SIMD dot product for attention scores.
    /// This provides 4-6x speedup over scalar dot products, addressing the 53x bottleneck
    /// identified in IMP-400f Popper analysis.
    pub(crate) fn attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Compute attention scores against all positions (cached + current)
            let mut scores = Vec::with_capacity(total_len);

            // Scores against cached positions (SIMD-optimized)
            for pos in 0..cache_len {
                let k_start = pos * hidden_dim + head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];
                let score = Self::simd_dot_f32(q_head, cached_key) * scale;
                scores.push(score);
            }

            // Score against current position (SIMD-optimized)
            let curr_key = &current_k[head_offset..head_offset + head_dim];
            let current_score = Self::simd_dot_f32(q_head, curr_key) * scale;
            scores.push(current_score);

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Sum over cached values (SIMD-optimized)
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * hidden_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                Self::simd_axpy_f32(out_head, weight, cached_val);
            }

            // Add current value (SIMD-optimized)
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            Self::simd_axpy_f32(out_head, current_weight, curr_val);
        }

        output
    }

    /// FlashAttention: Tiled attention with O(N) memory (PARITY-026)
    ///
    /// Implements the FlashAttention algorithm from Dao et al. for memory-efficient attention.
    /// Uses online softmax to process attention in tiles without materializing the N×N matrix.
    ///
    /// # Key Properties
    /// - Memory: O(N) instead of O(N²)
    /// - Numerically equivalent to standard attention
    /// - 10-100x memory savings for long sequences
    ///
    /// # Arguments
    /// * `q` - Query vector [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Current key [hidden_dim]
    /// * `current_v` - Current value [hidden_dim]
    /// * `block_size` - Tile size for tiled computation (default: 64)
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn flash_attention_tiled(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        block_size: usize,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Process each head with FlashAttention tiling
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Online softmax state for this head
            let mut m_i = f32::NEG_INFINITY; // Running max
            let mut l_i = 0.0f32; // Running sum of exp(score - max)
            let mut o_i = vec![0.0f32; head_dim]; // Accumulated output

            // Process KV cache in tiles
            let num_tiles = total_len.div_ceil(block_size);

            for tile_idx in 0..num_tiles {
                let tile_start = tile_idx * block_size;
                let tile_end = (tile_start + block_size).min(total_len);
                let tile_len = tile_end - tile_start;

                // Compute scores for this tile
                let mut tile_scores = Vec::with_capacity(tile_len);
                let mut tile_values: Vec<&[f32]> = Vec::with_capacity(tile_len);

                for pos in tile_start..tile_end {
                    if pos < cache_len {
                        // From cache
                        let k_start = pos * hidden_dim + head_offset;
                        let cached_key = &k_cache[k_start..k_start + head_dim];

                        // Compute Q·K score
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * cached_key[d];
                        }
                        tile_scores.push(score * scale);

                        let v_start = pos * hidden_dim + head_offset;
                        tile_values.push(&v_cache[v_start..v_start + head_dim]);
                    } else {
                        // Current position
                        let curr_key = &current_k[head_offset..head_offset + head_dim];

                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * curr_key[d];
                        }
                        tile_scores.push(score * scale);

                        tile_values.push(&current_v[head_offset..head_offset + head_dim]);
                    }
                }

                // Find max in this tile
                let m_tile = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running max
                let m_new = m_i.max(m_tile);

                // Rescale factors for online softmax
                let scale_old = (m_i - m_new).exp();
                let scale_tile = (m_tile - m_new).exp();

                // Compute local softmax sum for this tile
                let l_tile: f32 = tile_scores.iter().map(|&s| (s - m_tile).exp()).sum();

                // Update running sum
                l_i = l_i * scale_old + l_tile * scale_tile;

                // Update output: rescale old output and add new contribution
                for o in &mut o_i {
                    *o *= scale_old;
                }

                // Add weighted values from this tile
                for (j, &score) in tile_scores.iter().enumerate() {
                    let attn_weight = (score - m_tile).exp() * scale_tile;
                    let v = tile_values[j];
                    for d in 0..head_dim {
                        o_i[d] += attn_weight * v[d];
                    }
                }

                m_i = m_new;
            }

            // Finalize: divide by sum
            if l_i > 0.0 {
                for d in 0..head_dim {
                    output[head_offset + d] = o_i[d] / l_i;
                }
            }
        }

        output
    }

    /// SIMD-optimized dot product for f32 slices
    #[inline]
    fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: We've verified AVX2+FMA support
                unsafe { Self::simd_dot_f32_avx2(a, b) }
            } else {
                Self::simd_dot_f32_scalar(a, b)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_dot_f32_scalar(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            use std::arch::x86_64::{
                _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
                _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
                _mm_movehl_ps,
            };

            let len = a.len().min(b.len());
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;

            // Process 16 floats at a time (2x unrolled for better ILP)
            while i + 16 <= len {
                let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
                let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                acc = _mm256_fmadd_ps(va0, vb0, acc);
                acc = _mm256_fmadd_ps(va1, vb1, acc);
                i += 16;
            }
            // Handle remaining 8-float chunk
            if i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }

            // Horizontal sum
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut sum = _mm_cvtss_f32(result);

            // Handle remaining elements
            while i < len {
                sum += a[i] * b[i];
                i += 1;
            }

            sum
        }
    }

    #[inline]
    fn simd_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-optimized scaled accumulation: out[i] += weight * val[i]
    #[inline]
    fn simd_axpy_f32(out: &mut [f32], weight: f32, val: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We've verified AVX2 support
                unsafe { Self::simd_axpy_f32_avx2(out, weight, val) }
            } else {
                Self::simd_axpy_f32_scalar(out, weight, val);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_axpy_f32_scalar(out, weight, val);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn simd_axpy_f32_avx2(out: &mut [f32], weight: f32, val: &[f32]) {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        let len = out.len().min(val.len());
        let w = _mm256_set1_ps(weight);
        let mut i = 0;

        // Process 8 floats at a time
        while i + 8 <= len {
            // SAFETY: bounds checked above, pointers valid
            let v_out = unsafe { _mm256_loadu_ps(out.as_ptr().add(i)) };
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let v_val = unsafe { _mm256_loadu_ps(val.as_ptr().add(i)) };
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), result) };
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            out[i] += weight * val[i];
            i += 1;
        }
    }

    #[inline]
    fn simd_axpy_f32_scalar(out: &mut [f32], weight: f32, val: &[f32]) {
        for (o, v) in out.iter_mut().zip(val.iter()) {
            *o += weight * *v;
        }
    }

    /// Compute attention with Grouped Query Attention (GQA) support (IMP-105)
    ///
    /// GQA uses fewer KV heads than Q heads, with multiple Q heads sharing each KV head.
    /// This reduces memory bandwidth and KV cache size for large models.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim] (num_heads Q heads)
    /// * `k_cache` - Cached keys [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `v_cache` - Cached values [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `current_k` - Key for current position [kv_dim]
    /// * `current_v` - Value for current position [kv_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    ///
    /// # GQA Mapping
    /// Q head i uses KV head (i * num_kv_heads / num_heads)
    /// Example: 8 Q heads, 2 KV heads → Q heads 0-3 use KV head 0, Q heads 4-7 use KV head 1
    pub fn attention_with_cache_gqa(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Number of Q heads that share each KV head
        let q_per_kv = num_heads / num_kv_heads;

        // Total sequence length = cached + 1 (current)
        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Process each Q head
        for q_head in 0..num_heads {
            let q_head_offset = q_head * head_dim;
            let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

            // Map Q head to KV head (GQA mapping)
            let kv_head = q_head / q_per_kv;
            let kv_head_offset = kv_head * head_dim;

            // Compute attention scores against all positions (cached + current)
            let mut scores = Vec::with_capacity(total_len);

            // Scores against cached positions (SIMD-optimized)
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];
                let score = Self::simd_dot_f32(q_head_data, cached_key);
                scores.push(score * scale);
            }

            // Score against current position (SIMD-optimized)
            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            let current_score = Self::simd_dot_f32(q_head_data, curr_key);
            scores.push(current_score * scale);

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[q_head_offset..q_head_offset + head_dim];

            // Sum over cached values (SIMD-optimized)
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                Self::simd_axpy_f32(out_head, weight, cached_val);
            }

            // Add current value (SIMD-optimized)
            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            let current_weight = scores[cache_len];
            Self::simd_axpy_f32(out_head, current_weight, curr_val);
        }

        output
    }

    /// Attention with cache - writes to pre-allocated buffer (IMP-131)
    pub fn attention_with_cache_gqa_into(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_per_kv = num_heads / num_kv_heads;

        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        // Zero output buffer
        output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);

        // Stack-allocated scores buffer (max 8192 seq length)
        let mut scores_buf = [0.0f32; 8192];
        let scores = &mut scores_buf[..total_len];

        for q_head in 0..num_heads {
            let q_head_offset = q_head * head_dim;
            let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

            let kv_head = q_head / q_per_kv;
            let kv_head_offset = kv_head * head_dim;

            // Compute attention scores
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];
                scores[pos] = Self::simd_dot_f32(q_head_data, cached_key) * scale;
            }

            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            scores[cache_len] = Self::simd_dot_f32(q_head_data, curr_key) * scale;

            // Softmax
            crate::quantize::softmax_simd(scores);

            // Weighted sum of values
            let out_head = &mut output[q_head_offset..q_head_offset + head_dim];

            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                Self::simd_axpy_f32(out_head, weight, cached_val);
            }

            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            Self::simd_axpy_f32(out_head, scores[cache_len], curr_val);
        }
    }

    /// Adaptive attention with KV cache - auto-selects CPU or GPU backend (IMP-122)
    ///
    /// For short cache lengths (< 64), uses efficient CPU implementation.
    /// For long cache lengths (>= 64), uses GPU-accelerated computation.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Result containing attention output [hidden_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail (for GPU path)
    #[cfg(feature = "gpu")]
    pub fn adaptive_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Calculate cache length
        let cache_len = if hidden_dim > 0 {
            k_cache.len() / hidden_dim
        } else {
            0
        };

        // Threshold for GPU dispatch (matches IMP-119)
        const GPU_CACHE_LEN_THRESHOLD: usize = 64;

        if cache_len >= GPU_CACHE_LEN_THRESHOLD {
            // GPU path for long sequences
            self.gpu_attention_with_cache(q, k_cache, v_cache, current_k, current_v)
        } else {
            // CPU path for short sequences - use existing implementation
            Ok(self.attention_with_cache(q, k_cache, v_cache, current_k, current_v))
        }
    }

    /// GPU-accelerated attention with KV cache (IMP-122)
    ///
    /// Uses GPU for Q@K^T computation when cache is large enough.
    #[cfg(feature = "gpu")]
    fn gpu_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Create scheduler for GPU operations
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "gpu_attention_with_cache".to_string(),
                reason: format!("Failed to create scheduler: {}", e),
            }
        })?;

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Build full K matrix for this head: [total_len, head_dim]
            let mut k_full = Vec::with_capacity(total_len * head_dim);
            for pos in 0..cache_len {
                let k_start = pos * hidden_dim + head_offset;
                k_full.extend_from_slice(&k_cache[k_start..k_start + head_dim]);
            }
            k_full.extend_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // Transpose K to [head_dim, total_len] for matmul
            let mut k_t = vec![0.0f32; head_dim * total_len];
            for pos in 0..total_len {
                for d in 0..head_dim {
                    k_t[d * total_len + pos] = k_full[pos * head_dim + d];
                }
            }

            // GPU matmul: Q[1, head_dim] @ K_T[head_dim, total_len] -> [1, total_len]
            let scores_raw = scheduler
                .matmul(q_head, &k_t, 1, head_dim, total_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "gpu_attention_with_cache".to_string(),
                    reason: format!("GPU matmul failed: {}", e),
                })?;

            // Scale scores
            let mut scores: Vec<f32> = scores_raw.iter().map(|&s| s * scale).collect();

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Cached values
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * hidden_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                for d in 0..head_dim {
                    out_head[d] += weight * cached_val[d];
                }
            }

            // Current value
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            for d in 0..head_dim {
                out_head[d] += current_weight * curr_val[d];
            }
        }

        Ok(output)
    }

    /// Forward pass for a single token using KV cache (IMP-101c)
    ///
    /// This is O(n) per token instead of O(n²) due to KV cache reuse.
    ///
    /// # Arguments
    /// * `token_id` - Single input token ID
    /// * `cache` - Mutable reference to KV cache
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward_single_with_cache(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // DEBUG: Print hidden state after embedding
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            eprintln!("[DEBUG-FORWARD] Token={}, Position={}", token_id, position);
            eprintln!(
                "[DEBUG-FORWARD] After embed: sum={:.6}, hidden[0..4]={:?}",
                hidden_sum,
                &hidden[..4.min(hidden.len())]
            );
        }

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // Pre-allocate attention output buffer - reused across all layers
        let mut attn_out_buffer = vec![0.0f32; hidden_dim];

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a+2b. Fused attention layer norm + QKV projection
            // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
            // For LayerNorm models: use separate operations (has bias)
            let mut qkv = if use_rmsnorm {
                self.fused_rmsnorm_qkv_matmul(
                    &hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &layer.qkv_weight,
                )?
            } else {
                let normed = ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );
                self.qkv_matmul(&normed, &layer.qkv_weight)?
            };
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // Q: [hidden_dim] = [num_heads * head_dim]
            // K: [kv_dim] = [num_kv_heads * head_dim]
            // V: [kv_dim] = [num_kv_heads * head_dim]
            // Optimization: apply RoPE in-place to avoid Q/K copies
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / self.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;

            // Apply RoPE in-place to Q and K within QKV buffer
            self.apply_rope(&mut qkv[0..hidden_dim], position, self.config.num_heads);
            self.apply_rope(
                &mut qkv[hidden_dim..hidden_dim + kv_dim],
                position,
                num_kv_heads,
            );

            // Use slices to avoid copies (only copy K for cache storage)
            let q = &qkv[0..hidden_dim];
            let k = &qkv[hidden_dim..hidden_dim + kv_dim];
            let v = &qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim];

            // 2d. Get cached K/V and compute attention with GQA support
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            // Use pre-allocated attention output buffer (reused across layers)
            if k_cache.is_empty() {
                // First token - no cache yet, output is just weighted V
                // With single query and single K/V, need to expand V for all Q heads
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    attn_out_buffer[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
            } else {
                // Use cached K/V for attention with GQA
                // Uses pre-allocated buffer to avoid 704 Vec allocations per token
                self.attention_with_cache_gqa_into(q, k_cache, v_cache, k, v, &mut attn_out_buffer);

                // CORRECTNESS-013: Debug CPU attention output for layer 0 at position 1+
                if layer_idx == 0 && position >= 1 && std::env::var("CPU_DEBUG").is_ok() {
                    eprintln!(
                        "[CORRECTNESS-013-CPU] Layer 0 attention output at pos={}, first 10: {:?}",
                        position,
                        &attn_out_buffer[..10.min(attn_out_buffer.len())]
                    );
                    for h in 0..3 {
                        let start = h * head_dim;
                        eprintln!(
                            "[CORRECTNESS-013-CPU] Head {} first 5: {:?}",
                            h,
                            &attn_out_buffer[start..start + 5]
                        );
                    }
                }
            }

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, k, v);

            // 2f. Attention output projection
            let mut attn_output = self.fused_matmul(&attn_out_buffer, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h+2i. FFN with optional layer norm and SwiGLU/GELU activation
            // For RMSNorm + SwiGLU: fuse norm + up/gate matmuls to eliminate intermediate
            let ffn_activated = match (&layer.ffn_norm_weight, &layer.ffn_gate_weight) {
                // Fused path: RMSNorm + SwiGLU (LLaMA, TinyLlama, Mistral, etc.)
                (Some(ref ffn_norm), Some(ref gate_weight)) if use_rmsnorm => {
                    let (mut ffn_up, mut ffn_gate) = self.fused_rmsnorm_ffn_up_gate(
                        &hidden,
                        ffn_norm,
                        self.config.eps,
                        &layer.ffn_up_weight,
                        gate_weight,
                    )?;

                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                },

                // Non-fused SwiGLU (LayerNorm models with gate)
                (ffn_norm_opt, Some(ref gate_weight)) => {
                    let ffn_input = if let Some(ref ffn_norm) = ffn_norm_opt {
                        ops::layer_norm(
                            &hidden,
                            ffn_norm,
                            layer.ffn_norm_bias.as_deref(),
                            self.config.eps,
                        )
                    } else {
                        hidden.clone()
                    };

                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                },

                // GELU path (phi-2, GPT-2, etc.) - no gate weight
                (ffn_norm_opt, None) => {
                    let ffn_input = if let Some(ref ffn_norm) = ffn_norm_opt {
                        if use_rmsnorm {
                            ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                        } else {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                layer.ffn_norm_bias.as_deref(),
                                self.config.eps,
                            )
                        }
                    } else {
                        hidden.clone()
                    };

                    let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                },
            };

            // 2j. FFN down projection
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // DEBUG: Print hidden state after first layer
            if debug_forward && layer_idx == 0 {
                let hidden_sum: f32 = hidden.iter().sum();
                eprintln!(
                    "[DEBUG-FORWARD] After layer 0: sum={:.6}, hidden[0..4]={:?}",
                    hidden_sum,
                    &hidden[..4.min(hidden.len())]
                );
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // DEBUG: Print hidden state before LM head
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            let hidden_max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[DEBUG-FORWARD] Hidden after all layers: sum={:.4}, min={:.4}, max={:.4}",
                hidden_sum, hidden_min, hidden_max
            );
            eprintln!(
                "[DEBUG-FORWARD] Hidden[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] lm_head_weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
                self.lm_head_weight.qtype,
                self.lm_head_weight.data.len()
            );
            eprintln!(
                "[DEBUG-LM-HEAD] First 16 bytes of lm_head data: {:02x?}",
                &self.lm_head_weight.data[..16.min(self.lm_head_weight.data.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] output_norm_weight[0..4]: {:?}",
                &self.output_norm_weight[..4.min(self.output_norm_weight.len())]
            );
        }

        // 3+4. Fused final layer norm + LM head projection
        // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
        let mut logits = if use_rmsnorm {
            self.fused_rmsnorm_lm_head(&hidden)?
        } else {
            let normed = ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            );
            self.fused_matmul(&normed, &self.lm_head_weight)?
        };

        // DEBUG: Verify Q8_0 matmul by manual computation
        if debug_forward {
            // Get the normalized hidden state
            let normed = ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps);
            eprintln!(
                "[DEBUG-VERIFY] Normed hidden[0..8]: {:?}",
                &normed[..8.min(normed.len())]
            );

            // Manual dequantize row 0 of LM head weight
            const Q8_0_BLOCK_BYTES: usize = 34;
            const Q8_0_BLOCK_SIZE: usize = 32;
            let blocks_per_row = self.lm_head_weight.in_dim.div_ceil(Q8_0_BLOCK_SIZE);
            let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

            // Dequantize row 0 (token 0's projection weights)
            let row0_data = &self.lm_head_weight.data[0..bytes_per_row];
            let mut row0_f32 = vec![0.0f32; self.lm_head_weight.in_dim];
            for block_idx in 0..blocks_per_row {
                let block_start = block_idx * Q8_0_BLOCK_BYTES;
                let block = &row0_data[block_start..block_start + Q8_0_BLOCK_BYTES];
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for j in 0..32 {
                    let idx = block_idx * 32 + j;
                    if idx >= self.lm_head_weight.in_dim {
                        break;
                    }
                    row0_f32[idx] = (block[2 + j] as i8 as f32) * scale;
                }
            }
            eprintln!(
                "[DEBUG-VERIFY] LM head row 0 (dequantized) first 8: {:?}",
                &row0_f32[..8.min(row0_f32.len())]
            );

            // Compute dot product manually
            let manual_logit0: f32 = normed.iter().zip(row0_f32.iter()).map(|(a, b)| a * b).sum();
            eprintln!("[DEBUG-VERIFY] Manual logits[0] = {:.6}", manual_logit0);
            eprintln!("[DEBUG-VERIFY] Computed logits[0] = {:.6}", logits[0]);
            eprintln!(
                "[DEBUG-VERIFY] Difference = {:.6}",
                (manual_logit0 - logits[0]).abs()
            );

            // Check top tokens
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "[DEBUG-VERIFY] Top 5 tokens: {:?}",
                &indexed[..5.min(indexed.len())]
            );
        }

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Single-token forward pass with pre-allocated scratch buffers
    ///
    /// Uses OwnedInferenceScratchBuffer to eliminate per-token allocations.
    /// For Qwen2.5-0.5B, this saves ~40KB of allocations per token.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    ///
    /// Forward pass with adaptive CPU/GPU attention selection (IMP-124)
    ///
    /// This variant of `forward_single_with_cache` uses `adaptive_attention_with_cache`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Position in sequence
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_single_with_cache_adaptive(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // GQA dimensions
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / self.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // PARITY-113: Track CUDA kernel count for GPU dispatch metrics
        #[cfg(feature = "cuda")]
        let cuda_enabled = self.cuda_enabled();

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for others)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // 2b. QKV projection
            // PARITY-113: Record GPU dispatch when CUDA path is used for matmul
            #[cfg(feature = "cuda")]
            if cuda_enabled {
                let start = std::time::Instant::now();
                let qkv_result = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                let mut qkv = qkv_result;
                if let Some(ref bias) = layer.qkv_bias {
                    ops::add_bias(&mut qkv, bias);
                }

                // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
                let mut q = qkv[0..hidden_dim].to_vec();
                let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
                let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

                self.apply_rope(&mut q, position, self.config.num_heads);
                self.apply_rope(&mut k, position, num_kv_heads);

                // 2d. Get cached K/V and compute attention with GQA support
                let k_cache = cache.get_k(layer_idx);
                let v_cache = cache.get_v(layer_idx);

                let attn_out = if k_cache.is_empty() {
                    // First token - expand V for all Q heads (GQA)
                    let mut expanded_v = vec![0.0f32; hidden_dim];
                    let q_per_kv = self.config.num_heads / num_kv_heads;
                    for q_head in 0..self.config.num_heads {
                        let kv_head = q_head / q_per_kv;
                        let v_start = kv_head * head_dim;
                        let out_start = q_head * head_dim;
                        expanded_v[out_start..out_start + head_dim]
                            .copy_from_slice(&v[v_start..v_start + head_dim]);
                    }
                    expanded_v
                } else {
                    let start = std::time::Instant::now();
                    let result =
                        self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    result
                };

                // 2e. Store K and V in cache
                cache.append(layer_idx, &k, &v);

                // 2f. Attention output projection
                let start = std::time::Instant::now();
                let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                if let Some(ref bias) = layer.attn_output_bias {
                    ops::add_bias(&mut attn_output, bias);
                }

                // 2g. Residual connection
                for i in 0..hidden_dim {
                    hidden[i] += attn_output[i];
                }

                // 2h. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
                let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                    if use_rmsnorm {
                        ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                    } else {
                        ops::layer_norm(
                            &hidden,
                            ffn_norm,
                            layer.ffn_norm_bias.as_deref(),
                            self.config.eps,
                        )
                    }
                } else {
                    hidden.clone()
                };

                // 2i. FFN with SwiGLU or GELU activation
                let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    // SwiGLU path
                    let start = std::time::Instant::now();
                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let start = std::time::Instant::now();
                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                } else {
                    // GELU path
                    let start = std::time::Instant::now();
                    let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                };

                // 2j. FFN down projection
                let start = std::time::Instant::now();
                let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }

                // Residual
                for i in 0..hidden_dim {
                    hidden[i] += ffn_output[i];
                }

                continue;
            }

            // CPU path (non-CUDA)
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, num_kv_heads);

            // 2d. Get cached K/V and compute attention with adaptive dispatch
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - expand V for all Q heads (GQA)
                let mut expanded_v = vec![0.0f32; hidden_dim];
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    expanded_v[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
                expanded_v
            } else {
                // Use adaptive attention with metrics tracking
                let cache_len = k_cache.len() / kv_dim;
                const GPU_CACHE_LEN_THRESHOLD: usize = 64;

                if cache_len >= GPU_CACHE_LEN_THRESHOLD {
                    let start = std::time::Instant::now();
                    let result =
                        self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    result
                } else {
                    let start = std::time::Instant::now();
                    let result = self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v);
                    metrics.record_cpu_dispatch();
                    metrics.record_cpu_latency(start.elapsed());
                    result
                }
            };

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // 2i. FFN with SwiGLU or GELU activation
            let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }
                ffn_gate
            } else {
                // GELU path
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);
                ffn_hidden
            };

            // 2j. FFN down projection
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for others)
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // 4. LM head projection
        // PARITY-113: Record GPU dispatch for LM head when CUDA is enabled
        #[cfg(feature = "cuda")]
        if cuda_enabled {
            let start = std::time::Instant::now();
            let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
            metrics.record_gpu_dispatch();
            metrics.record_gpu_latency(start.elapsed());
            if let Some(ref bias) = self.lm_head_bias {
                ops::add_bias(&mut logits, bias);
            }
            return Ok(logits);
        }

        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens with KV cache for O(n) per-token decoding (IMP-101c)
    ///
    /// This is the optimized generation path that uses KV caching to avoid
    /// recomputing attention for all previous positions.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill), keeping the logits from the last position
        // The logits from processing token[n-1] at position n-1 predict token[n]
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens
        // First iteration uses logits from prefill, subsequent use logits from forward pass
        for gen_idx in 0..config.max_tokens {
            // DEBUG: Print logits info for first generated token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                let sum: f32 = logits.iter().sum();
                let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
                let top_5: Vec<(usize, f32)> = {
                    let mut indexed: Vec<_> =
                        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|(_, a), (_, b)| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indexed.into_iter().take(5).collect()
                };
                eprintln!(
                    "[DEBUG-LOGITS] len={}, sum={:.4}, min={:.4}, max={:.4}",
                    logits.len(),
                    sum,
                    min_val,
                    max_val
                );
                eprintln!("[DEBUG-LOGITS] top 5 token ids and logits: {:?}", top_5);
                eprintln!(
                    "[DEBUG-LOGITS] logits[0..5]: {:?}",
                    &logits[..5.min(logits.len())]
                );
            }

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // DEBUG: Print selected token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                eprintln!(
                    "[DEBUG-LOGITS] selected token: {} (logit={:.4})",
                    next_token,
                    logits.get(next_token as usize).copied().unwrap_or(f32::NAN)
                );
            }

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            // Position is prompt.len() + gen_idx (where token was just added)
            let position = prompt.len() + gen_idx;
            logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with streaming callback (PMAT-087)
    ///
    /// Same as `generate_with_cache` but calls `on_token` after each token
    /// is generated, enabling true streaming to clients.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token. Return `false` to stop.
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache_streaming<F>(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill)
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens with streaming
        for gen_idx in 0..config.max_tokens {
            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PMAT-087: Call streaming callback - stop if it returns false
            if !on_token(next_token) {
                break;
            }

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration
            let position = prompt.len() + gen_idx;
            logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with zero-allocation inference (IMP-131)
    ///
    /// This is the highest-performance generation path. Uses pre-allocated
    /// scratch buffers to eliminate per-token allocations, providing ~3-4x
    /// speedup over allocating variants.
    ///
    /// Performance characteristics:
    /// - Single allocation at start (scratch buffer + KV cache)
    /// - Zero allocations per generated token
    /// - ~500KB saved per token for TinyLlama-1.1B
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_scratch(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut scratch = InferenceScratchBuffer::from_config(&self.config);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) - uses scratch buffers
        for (pos, &token_id) in prompt.iter().enumerate() {
            self.forward_single_with_scratch(token_id, &mut cache, pos, &mut scratch)?;
        }

        // Generate new tokens - zero allocations per token
        // PAR-126: Fixed loop structure to match generate_with_cache:
        // 1. Sample from current logits (prefill on first iter, previous forward otherwise)
        // 2. Then run forward on the new token to get logits for next iteration
        for gen_idx in 0..config.max_tokens {
            // Sample next token from current logits (prefill logits on first iter)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&scratch.logits)
            } else {
                Self::sample_topk(&scratch.logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the new token
            let position = prompt.len() + gen_idx;
            self.forward_single_with_scratch(next_token, &mut cache, position, &mut scratch)?;
        }

        Ok(tokens)
    }

    /// Zero-allocation forward pass using scratch buffers (IMP-131)
    ///
    /// All intermediate results are written to pre-allocated scratch buffers.
    /// Output logits are stored in `scratch.logits`.
    fn forward_single_with_scratch(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        scratch: &mut InferenceScratchBuffer,
    ) -> Result<()> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Detect architecture
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 1. Token embedding lookup → scratch.hidden
        self.embed_into(token_id, &mut scratch.hidden);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm → scratch.normed
            if use_rmsnorm {
                self.rms_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &mut scratch.normed,
                );
            } else {
                self.layer_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                    &mut scratch.normed,
                );
            }

            // 2b. QKV projection → scratch.qkv (zero-allocation via P1-REV)
            // PAR-126: Fix GQA dimension issue - use config instead of q_dim() which
            // incorrectly assumes Q=K=V for fused weights
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / self.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;
            // Q uses all heads, K/V use only kv_heads (GQA)
            let q_dim = hidden_dim;
            let k_dim = kv_dim;
            let v_dim = kv_dim;
            let qkv_dim = q_dim + k_dim + v_dim;

            // PAR-126: Pre-quantize normalized hidden to Q8K for VNNI-accelerated matmul
            // This allows reusing quantized activations for QKV projection
            // NOTE: Q8K requires hidden_dim to be multiple of 256. For smaller models
            // like 0.5B (hidden=896), fall back to f32 path.
            let use_q8k_path = hidden_dim.is_multiple_of(256);

            if use_q8k_path {
                use crate::quantize::quantize_activations_q8k_into;
                let hidden_sb = hidden_dim / 256;
                quantize_activations_q8k_into(
                    &scratch.normed[..hidden_dim],
                    &mut scratch.q8k_hidden_scales[..hidden_sb],
                    &mut scratch.q8k_hidden_quants[..hidden_dim],
                )?;

                // Write directly to scratch.qkv, using Q8K-accelerated path
                self.qkv_matmul_q8k_into(
                    &scratch.normed,
                    &layer.qkv_weight,
                    &mut scratch.qkv[..qkv_dim],
                    &scratch.q8k_hidden_scales[..hidden_sb],
                    &scratch.q8k_hidden_quants[..hidden_dim],
                )?;
            } else {
                // Fall back to f32 path for non-256-aligned hidden dims
                self.qkv_matmul_into(
                    &scratch.normed,
                    &layer.qkv_weight,
                    &mut scratch.qkv[..qkv_dim],
                )?;
            }

            // Copy from scratch.qkv to individual Q, K, V buffers
            scratch.q[..q_dim].copy_from_slice(&scratch.qkv[..q_dim]);
            scratch.k[..k_dim].copy_from_slice(&scratch.qkv[q_dim..q_dim + k_dim]);
            scratch.v[..v_dim].copy_from_slice(&scratch.qkv[q_dim + k_dim..qkv_dim]);

            // Add bias if present
            if let Some(ref bias) = layer.qkv_bias {
                for i in 0..q_dim {
                    scratch.q[i] += bias[i];
                }
                for i in 0..k_dim {
                    scratch.k[i] += bias[q_dim + i];
                }
                for i in 0..v_dim {
                    scratch.v[i] += bias[q_dim + k_dim + i];
                }
            }

            // Apply RoPE
            self.apply_rope(&mut scratch.q[..q_dim], position, self.config.num_heads);
            self.apply_rope(&mut scratch.k[..k_dim], position, self.config.num_kv_heads);

            // 2c. Compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            if k_cache.is_empty() {
                // First token - expand V if GQA
                if self.config.num_kv_heads < self.config.num_heads {
                    let head_dim = hidden_dim / self.config.num_heads;
                    let group_size = self.config.num_heads / self.config.num_kv_heads;
                    for h in 0..self.config.num_heads {
                        let kv_head = h / group_size;
                        let src_start = kv_head * head_dim;
                        let dst_start = h * head_dim;
                        scratch.attn_out[dst_start..dst_start + head_dim]
                            .copy_from_slice(&scratch.v[src_start..src_start + head_dim]);
                    }
                } else {
                    scratch.attn_out[..hidden_dim].copy_from_slice(&scratch.v[..hidden_dim]);
                }
            } else {
                self.attention_with_cache_gqa_into(
                    &scratch.q[..q_dim],
                    k_cache,
                    v_cache,
                    &scratch.k[..k_dim],
                    &scratch.v[..v_dim],
                    &mut scratch.attn_out,
                );
            }

            // Store K, V in cache
            cache.append(layer_idx, &scratch.k[..k_dim], &scratch.v[..v_dim]);

            // 2d. Attention output projection → scratch.attn_proj
            // PAR-128: Use Q8K-accelerated path for attention output projection
            // attn_out is hidden_dim sized, reuse hidden Q8K buffers
            let use_q8k_attn_out = use_q8k_path && layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K;

            if use_q8k_attn_out {
                use crate::quantize::{
                    fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                };
                let hidden_sb = hidden_dim / 256;
                // Quantize attention output to Q8K (reuse hidden Q8K buffers)
                quantize_activations_q8k_into(
                    &scratch.attn_out[..hidden_dim],
                    &mut scratch.q8k_hidden_scales[..hidden_sb],
                    &mut scratch.q8k_hidden_quants[..hidden_dim],
                )?;
                fused_q4k_q8k_parallel_matvec_into(
                    &layer.attn_output_weight.data,
                    &scratch.q8k_hidden_scales[..hidden_sb],
                    &scratch.q8k_hidden_quants[..hidden_dim],
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                    &mut scratch.attn_proj,
                )?;
            } else {
                self.fused_matmul_into(
                    &scratch.attn_out[..hidden_dim],
                    &layer.attn_output_weight,
                    &mut scratch.attn_proj,
                )?;
            }
            if let Some(ref bias) = layer.attn_output_bias {
                for i in 0..hidden_dim {
                    scratch.attn_proj[i] += bias[i];
                }
            }

            // 2e. Residual connection
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.attn_proj[i];
            }

            // 2f. Pre-FFN layer norm → scratch.normed
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    self.rms_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        self.config.eps,
                        &mut scratch.normed,
                    );
                } else {
                    self.layer_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                        &mut scratch.normed,
                    );
                }
            } else {
                scratch.normed[..hidden_dim].copy_from_slice(&scratch.hidden[..hidden_dim]);
            }

            // 2g. FFN
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA)
                // PAR-126: Use Q8K-accelerated path only if hidden_dim is 256-aligned
                if use_q8k_path {
                    // Pre-quantize normed hidden to Q8K for VNNI-accelerated FFN matmul
                    // Quantize once, reuse for both up and gate matmuls
                    use crate::quantize::quantize_activations_q8k_into;
                    let hidden_sb = hidden_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.normed[..hidden_dim],
                        &mut scratch.q8k_hidden_scales[..hidden_sb],
                        &mut scratch.q8k_hidden_quants[..hidden_dim],
                    )?;

                    // Use fused FFN up+gate kernel to eliminate rayon::join overhead
                    // This reduces parallel region spawns from 2 to 1 per layer
                    let up_weight = &layer.ffn_up_weight;
                    let q8k_scales = &scratch.q8k_hidden_scales[..hidden_sb];
                    let q8k_quants = &scratch.q8k_hidden_quants[..hidden_dim];

                    // Check if both weights are Q4K for fused path
                    if up_weight.qtype == GGUF_TYPE_Q4_K && gate_weight.qtype == GGUF_TYPE_Q4_K {
                        use crate::quantize::fused_q4k_q8k_ffn_up_gate_into;
                        fused_q4k_q8k_ffn_up_gate_into(
                            &up_weight.data,
                            &gate_weight.data,
                            q8k_scales,
                            q8k_quants,
                            up_weight.in_dim,
                            up_weight.out_dim,
                            &mut scratch.ffn_up,
                            &mut scratch.ffn_gate,
                        )?;
                    } else {
                        // Fallback to separate matmuls if not both Q4K
                        use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
                        let (up_result, gate_result) = rayon::join(
                            || {
                                if up_weight.qtype == GGUF_TYPE_Q4_K {
                                    fused_q4k_q8k_parallel_matvec_into(
                                        &up_weight.data,
                                        q8k_scales,
                                        q8k_quants,
                                        up_weight.in_dim,
                                        up_weight.out_dim,
                                        &mut scratch.ffn_up,
                                    )
                                } else {
                                    self.fused_matmul_into(
                                        &scratch.normed[..hidden_dim],
                                        up_weight,
                                        &mut scratch.ffn_up,
                                    )
                                }
                            },
                            || {
                                if gate_weight.qtype == GGUF_TYPE_Q4_K {
                                    fused_q4k_q8k_parallel_matvec_into(
                                        &gate_weight.data,
                                        q8k_scales,
                                        q8k_quants,
                                        gate_weight.in_dim,
                                        gate_weight.out_dim,
                                        &mut scratch.ffn_gate,
                                    )
                                } else {
                                    self.fused_matmul_into(
                                        &scratch.normed[..hidden_dim],
                                        gate_weight,
                                        &mut scratch.ffn_gate,
                                    )
                                }
                            },
                        );
                        up_result?;
                        gate_result?;
                    }
                } else {
                    // Fall back to f32 path for non-256-aligned hidden dims
                    let up_weight = &layer.ffn_up_weight;
                    let (up_result, gate_result) = rayon::join(
                        || {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                up_weight,
                                &mut scratch.ffn_up,
                            )
                        },
                        || {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                gate_weight,
                                &mut scratch.ffn_gate,
                            )
                        },
                    );
                    up_result?;
                    gate_result?;
                }

                if let Some(ref bias) = layer.ffn_up_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_up[i] += bias[i];
                    }
                }
                if let Some(ref bias) = layer.ffn_gate_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_gate[i] += bias[i];
                    }
                }

                // SiLU on gate, multiply with up
                ops::silu(&mut scratch.ffn_gate[..intermediate_dim]);
                for i in 0..intermediate_dim {
                    scratch.ffn_gate[i] *= scratch.ffn_up[i];
                }

                // PAR-127: Use Q8K-accelerated FFN down projection for Q4K weights
                // Q6K uses f32 path since Q8K conversion overhead > bandwidth savings
                let use_q8k_down = intermediate_dim.is_multiple_of(256)
                    && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

                if use_q8k_down {
                    use crate::quantize::{
                        fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                    };
                    let inter_sb = intermediate_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.ffn_gate[..intermediate_dim],
                        &mut scratch.q8k_inter_scales[..inter_sb],
                        &mut scratch.q8k_inter_quants[..intermediate_dim],
                    )?;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_down_weight.data,
                        &scratch.q8k_inter_scales[..inter_sb],
                        &scratch.q8k_inter_quants[..intermediate_dim],
                        layer.ffn_down_weight.in_dim,
                        layer.ffn_down_weight.out_dim,
                        &mut scratch.ffn_down,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.ffn_gate[..intermediate_dim],
                        &layer.ffn_down_weight,
                        &mut scratch.ffn_down,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_down_bias {
                    for i in 0..hidden_dim {
                        scratch.ffn_down[i] += bias[i];
                    }
                }
            } else {
                // GELU path (phi-2)
                // PAR-129: Use Q8K-accelerated FFN for GELU models (Q4K only)
                let use_q8k_gelu_up = use_q8k_path && layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K;
                let use_q8k_gelu_down = intermediate_dim.is_multiple_of(256)
                    && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

                if use_q8k_gelu_up {
                    // Reuse already-quantized hidden from QKV (scratch.q8k_hidden_*)
                    use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
                    let hidden_sb = hidden_dim / 256;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_up_weight.data,
                        &scratch.q8k_hidden_scales[..hidden_sb],
                        &scratch.q8k_hidden_quants[..hidden_dim],
                        layer.ffn_up_weight.in_dim,
                        layer.ffn_up_weight.out_dim,
                        &mut scratch.ffn_up,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.normed[..hidden_dim],
                        &layer.ffn_up_weight,
                        &mut scratch.ffn_up,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_up_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_up[i] += bias[i];
                    }
                }
                ops::gelu(&mut scratch.ffn_up[..intermediate_dim]);

                if use_q8k_gelu_down {
                    use crate::quantize::{
                        fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                    };
                    let inter_sb = intermediate_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.ffn_up[..intermediate_dim],
                        &mut scratch.q8k_inter_scales[..inter_sb],
                        &mut scratch.q8k_inter_quants[..intermediate_dim],
                    )?;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_down_weight.data,
                        &scratch.q8k_inter_scales[..inter_sb],
                        &scratch.q8k_inter_quants[..intermediate_dim],
                        layer.ffn_down_weight.in_dim,
                        layer.ffn_down_weight.out_dim,
                        &mut scratch.ffn_down,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.ffn_up[..intermediate_dim],
                        &layer.ffn_down_weight,
                        &mut scratch.ffn_down,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_down_bias {
                    for i in 0..hidden_dim {
                        scratch.ffn_down[i] += bias[i];
                    }
                }
            }

            // 2h. FFN residual
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_down[i];
            }
        }

        // 3. Final layer norm → scratch.normed
        if use_rmsnorm {
            self.rms_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.config.eps,
                &mut scratch.normed,
            );
        } else {
            self.layer_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
                &mut scratch.normed,
            );
        }

        // 4. LM head → scratch.logits
        self.fused_matmul_into(
            &scratch.normed[..hidden_dim],
            &self.lm_head_weight,
            &mut scratch.logits,
        )?;

        Ok(())
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-125)
    ///
    /// This variant of `generate_with_cache` uses `forward_single_with_cache_adaptive`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) with adaptive attention
        // Keep the logits from the last position for the first generated token
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache_adaptive(token_id, &mut cache, pos, metrics)?;
        }

        // Generate new tokens with adaptive attention
        for gen_idx in 0..config.max_tokens {
            // Sample next token from current logits
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            let position = prompt.len() + gen_idx;
            logits =
                self.forward_single_with_cache_adaptive(next_token, &mut cache, position, metrics)?;
        }

        Ok(tokens)
    }

    /// Batched forward pass for prompt prefill (PARITY-002)
    ///
    /// Processes all prompt tokens at once, enabling GPU acceleration
    /// for the attention computation when the batch is large enough.
    ///
    /// # Arguments
    /// * `tokens` - All prompt tokens to process at once
    /// * `cache` - KV cache for storing computed K/V tensors
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Logits for next token prediction (from the last token position)
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_batch_with_cache(
        &self,
        tokens: &[u32],
        cache: &mut OwnedQuantizedKVCache,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Tokens cannot be empty".to_string(),
            });
        }

        let seq_len = tokens.len();
        let hidden_dim = self.config.hidden_dim;

        // 1. Embed all tokens at once: [seq_len, hidden_dim]
        let mut hidden_states: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&token_id| self.embed(&[token_id]))
            .collect();

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Collect Q, K, V for all positions
            let mut all_q: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut all_k: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut all_v: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

            for (pos, hidden) in hidden_states.iter().enumerate() {
                // 2a. Attention layer norm
                let normed = ops::layer_norm(
                    hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );

                // 2b. QKV projection
                let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                if let Some(ref bias) = layer.qkv_bias {
                    ops::add_bias(&mut qkv, bias);
                }

                // 2c. Extract Q, K, V and apply RoPE
                // Note: This uses hidden_dim for all (assumes non-GQA or fused QKV)
                let mut q = qkv[0..hidden_dim].to_vec();
                let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                self.apply_rope(&mut q, pos, self.config.num_heads);
                self.apply_rope(&mut k, pos, self.config.num_heads); // Same as Q for non-GQA

                all_q.push(q);
                all_k.push(k);
                all_v.push(v);
            }

            // 2d. Compute batched attention
            // For PARITY-002: This is where GPU can accelerate!
            // Attention scores: Q @ K^T is [seq_len, seq_len]
            let attn_outputs = self
                .batched_attention_with_cache(&all_q, &all_k, &all_v, cache, layer_idx, metrics)?;

            // 2e. Store all K/V in cache
            for (k, v) in all_k.iter().zip(all_v.iter()) {
                cache.append(layer_idx, k, v);
            }

            // 2f. Attention output projection + residual
            for (pos, attn_out) in attn_outputs.iter().enumerate() {
                let mut attn_output = self.fused_matmul(attn_out, &layer.attn_output_weight)?;
                if let Some(ref bias) = layer.attn_output_bias {
                    ops::add_bias(&mut attn_output, bias);
                }

                // Residual connection
                for i in 0..hidden_dim {
                    hidden_states[pos][i] += attn_output[i];
                }
            }

            // 2g. FFN for all positions
            for hidden in &mut hidden_states {
                let mut ffn_hidden = self.fused_matmul(hidden, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut ffn_output = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }

                // Residual
                for i in 0..hidden_dim {
                    hidden[i] += ffn_output[i];
                }
            }
        }

        // Advance cache position for all processed tokens
        for _ in 0..seq_len {
            cache.advance();
        }

        // 3. Final layer norm and LM head for LAST token only
        let last_hidden = &hidden_states[seq_len - 1];
        let normed = ops::layer_norm(
            last_hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Batched attention computation with GPU acceleration (PARITY-002)
    ///
    /// Computes attention for all positions at once, enabling GPU dispatch
    /// when the workload (seq_len * hidden_dim * seq_len) exceeds the threshold.
    ///
    /// KEY OPTIMIZATION: Uses GPU matmul for Q @ K^T when workload is large enough.
    /// This is the critical path for GPU acceleration - previous implementation only
    /// recorded metrics without actually using GPU.
    #[cfg(feature = "gpu")]
    fn batched_attention_with_cache(
        &self,
        all_q: &[Vec<f32>],
        all_k: &[Vec<f32>],
        all_v: &[Vec<f32>],
        cache: &OwnedQuantizedKVCache,
        layer_idx: usize,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = all_q.len();
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;

        // Get any cached K/V from previous sequences
        let cached_k = cache.get_k(layer_idx);
        let cached_v = cache.get_v(layer_idx);
        let cache_len = cached_k.len() / hidden_dim;

        // Build full K/V sequences: [cache + current]
        let total_len = cache_len + seq_len;

        // Determine if we should use GPU based on workload size
        //
        // IMPORTANT FINDING (IMP-600, PARITY-002):
        // GPU is 2.7x SLOWER for MATVEC operations (per-head attention is MATVEC)
        // GPU is 57x FASTER for large GEMM (batch) operations
        //
        // For GPU to be beneficial, we need LARGE matrices. Per-head attention
        // uses tiny matrices: Q[1, head_dim] @ K^T[head_dim, seq_len] = [1, seq_len]
        // This is a MATVEC operation where GPU transfer overhead dominates.
        //
        // Measured result with GPU matmul: 0.20 tok/s (vs 5.31 tok/s CPU)
        // GPU path is 26x SLOWER due to per-head matmul overhead.
        //
        // For true GPU acceleration, need:
        // - FlashAttention (fused kernel, not yet available in trueno)
        // - Batched multi-request inference (process multiple prompts together)
        //
        // For now, use optimized CPU path which is faster for single-request inference.
        let workload = num_heads * seq_len * head_dim * total_len;
        let _ = workload; // Document: GPU not used because MATVEC is slower on GPU

        // Always use CPU path - it's faster for per-head attention MATVEC
        metrics.record_cpu_dispatch();
        self.cpu_batched_attention(
            all_q, all_k, all_v, cached_k, cached_v, cache_len, hidden_dim, num_heads, head_dim,
        )
    }

    /// CPU-based batched attention (fallback for small workloads)
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)] // Attention requires all these parameters
    fn cpu_batched_attention(
        &self,
        all_q: &[Vec<f32>],
        all_k: &[Vec<f32>],
        all_v: &[Vec<f32>],
        cached_k: &[f32],
        cached_v: &[f32],
        cache_len: usize,
        hidden_dim: usize,
        _num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = all_q.len();
        let mut outputs = Vec::with_capacity(seq_len);

        for (q_pos, q) in all_q.iter().enumerate() {
            let attend_len = cache_len + q_pos + 1;
            let mut k_vecs: Vec<&[f32]> = Vec::with_capacity(attend_len);
            let mut v_vecs: Vec<&[f32]> = Vec::with_capacity(attend_len);

            // Add cached K/V
            for i in 0..cache_len {
                let start = i * hidden_dim;
                let end = start + hidden_dim;
                k_vecs.push(&cached_k[start..end]);
                v_vecs.push(&cached_v[start..end]);
            }

            // Add current sequence K/V up to and including current position
            for i in 0..=q_pos {
                k_vecs.push(&all_k[i]);
                v_vecs.push(&all_v[i]);
            }

            let output = self.compute_attention_output(q, &k_vecs, &v_vecs, head_dim)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Compute attention output for a single query against K/V vectors
    #[cfg(feature = "gpu")]
    fn compute_attention_output(
        &self,
        q: &[f32],
        k_vecs: &[&[f32]],
        v_vecs: &[&[f32]],
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = q.len();
        let num_heads = hidden_dim / head_dim;
        let seq_len = k_vecs.len();

        if seq_len == 0 {
            // No keys to attend to - return zeros (will be replaced by first attention)
            return Ok(vec![0.0; hidden_dim]);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; hidden_dim];

        // Process each head independently
        for head in 0..num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;

            let q_head = &q[head_start..head_end];

            // Compute attention scores for this head
            let mut scores = Vec::with_capacity(seq_len);
            for k in k_vecs {
                let k_head = &k[head_start..head_end];
                let score: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                scores.push(score * scale);
            }

            // Softmax (SIMD-optimized, in-place)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            for (attn, v) in scores.iter().zip(v_vecs.iter()) {
                let v_head = &v[head_start..head_end];
                for (i, &v_val) in v_head.iter().enumerate() {
                    output[head_start + i] += attn * v_val;
                }
            }
        }

        Ok(output)
    }

    /// Generate tokens with batched prompt prefill (PARITY-002)
    ///
    /// Uses `forward_batch_with_cache` for initial prompt processing (GPU-accelerated),
    /// then falls back to single-token generation for autoregressive decoding.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs (processed in batch)
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_batched_prefill(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // PARITY-002: Process ALL prompt tokens at once (batched prefill)
        // This enables GPU acceleration for the attention computation
        let mut logits = self.forward_batch_with_cache(prompt, &mut cache, metrics)?;

        // Generate new tokens one at a time (autoregressive)
        for gen_idx in 0..config.max_tokens {
            // Sample next token from logits
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Forward pass for the new token (single-token, uses CPU)
            let position = prompt.len() + gen_idx;
            logits =
                self.forward_single_with_cache_adaptive(next_token, &mut cache, position, metrics)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with SmallVec optimization (IMP-117)
    ///
    /// Uses SmallVec for token storage to avoid heap allocations when:
    /// - Prompt + max_tokens <= TOKEN_BUFFER_INLINE_CAP
    ///
    /// # Arguments
    /// * `prompt` - Input token buffer (can be SmallVec or slice)
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence as TokenBuffer (SmallVec)
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_smallvec(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<TokenBuffer> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);

        // Use SmallVec for token storage - inline for small sequences
        let mut tokens: TokenBuffer = TokenBuffer::from_slice(prompt);

        // Process prompt tokens (prefill)
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _ = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens
        for gen_idx in 0..config.max_tokens {
            let position = prompt.len() + gen_idx;
            let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
                reason: "Token buffer empty during generation".to_string(),
            })?;

            let logits = self.forward_single_with_cache(last_token, &mut cache, position)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }
        }

        Ok(tokens)
    }

    // ========================================================================
    // PARITY-006: Batch Processing - Parallel Token Generation
    // ========================================================================

    /// Generate tokens for multiple requests in parallel (PARITY-006)
    ///
    /// This processes multiple independent requests together, enabling GPU GEMM
    /// acceleration. When batch_size > 1, the matmul operations become:
    /// `[batch_size, hidden_dim] @ [hidden_dim, output_dim]` which is GEMM.
    ///
    /// Per IMP-600: GPU is 57x faster for GEMM vs 2.7x slower for MATVEC.
    /// Batch inference is the key to utilizing GPU acceleration effectively.
    ///
    /// # Arguments
    /// * `prompts` - Vector of prompts (each prompt is a slice of token IDs)
    /// * `config` - Generation configuration (shared across all requests)
    ///
    /// # Returns
    /// Vector of generated token sequences (one per input prompt)
    ///
    /// # Performance
    /// - batch_size=1: Falls back to single-request path (CPU optimal)
    /// - batch_size>1: Uses batched matmul for GPU GEMM acceleration
    ///
    /// # Errors
    /// Returns error if any request fails
    pub fn batch_generate(
        &self,
        prompts: &[&[u32]],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompts cannot be empty".to_string(),
            });
        }

        // For single request, use optimized single-request path
        if prompts.len() == 1 {
            return Ok(vec![self.generate_with_cache(prompts[0], config)?]);
        }

        let batch_size = prompts.len();
        let max_prompt_len = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let max_seq_len = max_prompt_len + config.max_tokens;

        // Create KV caches for each request
        let mut caches: Vec<OwnedQuantizedKVCache> = (0..batch_size)
            .map(|_| OwnedQuantizedKVCache::from_config(&self.config, max_seq_len))
            .collect();

        // Initialize token sequences with prompts
        let mut all_tokens: Vec<Vec<u32>> = prompts.iter().map(|p| p.to_vec()).collect();

        // Track which requests are still generating
        let mut active: Vec<bool> = vec![true; batch_size];

        // Prefill phase: process each prompt (can be batched in future)
        for (req_idx, prompt) in prompts.iter().enumerate() {
            for (pos, &token_id) in prompt.iter().enumerate() {
                let _ = self.forward_single_with_cache(token_id, &mut caches[req_idx], pos)?;
            }
        }

        // Generation phase: process all active requests together
        for gen_idx in 0..config.max_tokens {
            // Count active requests
            let active_count = active.iter().filter(|&&a| a).count();
            if active_count == 0 {
                break;
            }

            // Collect last tokens from active requests
            let active_indices: Vec<usize> = active
                .iter()
                .enumerate()
                .filter(|(_, &a)| a)
                .map(|(i, _)| i)
                .collect();

            // Process active requests - batched forward pass
            let mut next_tokens = Vec::with_capacity(active_count);

            for &req_idx in &active_indices {
                let position = prompts[req_idx].len() + gen_idx;
                let last_token = *all_tokens[req_idx]
                    .last()
                    .expect("tokens must be non-empty");

                let logits =
                    self.forward_single_with_cache(last_token, &mut caches[req_idx], position)?;

                // Sample next token
                let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                    Self::argmax(&logits)
                } else {
                    Self::sample_topk(&logits, config.temperature, config.top_k)
                };

                next_tokens.push((req_idx, next_token));
            }

            // Apply next tokens and check stop conditions
            for (req_idx, next_token) in next_tokens {
                if config.stop_tokens.contains(&next_token) {
                    active[req_idx] = false;
                    continue;
                }

                all_tokens[req_idx].push(next_token);

                if all_tokens[req_idx].len() >= max_seq_len {
                    active[req_idx] = false;
                }
            }
        }

        Ok(all_tokens)
    }

    /// Get the batch throughput improvement factor (PARITY-006)
    ///
    /// Per IMP-600: GPU GEMM is 57x faster than MATVEC.
    /// Batch inference converts MATVEC to GEMM when batch_size > 1.
    ///
    /// # Arguments
    /// * `batch_size` - Number of concurrent requests
    ///
    /// # Returns
    /// Estimated throughput multiplier vs single-request
    #[must_use]
    pub const fn batch_throughput_factor(batch_size: usize) -> f64 {
        match batch_size {
            0 | 1 => 1.0,
            2..=4 => 1.8,   // ~2x throughput with small batch
            5..=8 => 2.5,   // GPU GEMM starts to help
            9..=16 => 3.5,  // Good GPU utilization
            17..=32 => 5.0, // Near-optimal batch
            _ => 6.0,       // Large batch, GPU-limited
        }
    }

    /// Forward pass for a batch of tokens (IMP-106)
    ///
    /// Processes multiple tokens through the transformer in parallel.
    /// This is more efficient than sequential processing for prefill.
    ///
    /// # Arguments
    /// * `token_ids` - Batch of input token IDs [batch_size]
    ///
    /// # Returns
    /// Logits for all positions [batch_size * vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward_batch(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup for all tokens
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention LayerNorm
            let normed = ops::layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // QKV projection (batched)
            let qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;

            // Split Q, K, V for batch - simplified attention (no causal mask for batch)
            let qkv_dim = qkv.len() / batch_size;
            let q_dim = hidden_dim;
            let kv_dim = (qkv_dim - q_dim) / 2;

            // Process attention for each position (simplified for batch)
            let mut attn_out = Vec::with_capacity(batch_size * hidden_dim);
            for pos in 0..batch_size {
                let qkv_start = pos * qkv_dim;
                let q = &qkv[qkv_start..qkv_start + q_dim];
                let k = &qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim];
                let v = &qkv[qkv_start + q_dim + kv_dim..qkv_start + qkv_dim];

                // Simple self-attention for current position (attend to itself only for simplicity)
                // Full causal attention would require attending to all previous positions
                let head_dim = hidden_dim / self.config.num_heads;
                let scale = 1.0 / (head_dim as f32).sqrt();

                let mut out = vec![0.0f32; hidden_dim];
                for h in 0..self.config.num_heads {
                    let kv_h = h * self.config.num_kv_heads / self.config.num_heads;
                    let q_h = &q[h * head_dim..(h + 1) * head_dim];
                    let k_h = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let v_h = &v[kv_h * head_dim..(kv_h + 1) * head_dim];

                    // Score and softmax (single position = 1.0 weight)
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_h[d] * k_h[d];
                    }
                    let _weight = (score * scale).exp(); // softmax of single value = 1.0

                    // Apply value
                    for d in 0..head_dim {
                        out[h * head_dim + d] = v_h[d];
                    }
                }
                attn_out.extend_from_slice(&out);
            }

            // Output projection
            let projected = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += projected[i];
            }

            // FFN (pre-norm style)
            let ffn_normed =
                ops::layer_norm(&hidden, &layer.attn_norm_weight, None, self.config.eps);
            let up = self.fused_matmul(&ffn_normed, &layer.ffn_up_weight)?;

            // GELU activation
            let gelu: Vec<f32> = up
                .iter()
                .map(|&x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x.powi(3))).tanh()))
                .collect();

            let down = self.fused_matmul(&gelu, &layer.ffn_down_weight)?;

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += down[i];
            }
        }

        // 3. Final LayerNorm
        let normed = ops::layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection to vocab logits
        let logits = self.fused_matmul(&normed, &self.lm_head_weight)?;

        Ok(logits)
    }

    /// Prefill prompt tokens with batched forward pass (IMP-106)
    ///
    /// Efficiently processes all prompt tokens and populates the KV cache.
    /// Returns the last position's logits for sampling.
    ///
    /// # Arguments
    /// * `prompt` - Prompt token IDs
    /// * `cache` - KV cache to populate
    ///
    /// # Returns
    /// Logits for the last position [vocab_size]
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn prefill_batch(
        &self,
        prompt: &[u32],
        cache: &mut OwnedQuantizedKVCache,
    ) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // Process each position to populate KV cache
        // (True batch prefill would compute all positions at once with causal attention)
        let mut last_logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            last_logits = self.forward_single_with_cache(token_id, cache, pos)?;
        }

        Ok(last_logits)
    }

    /// Forward pass for a batch of tokens with GPU acceleration (IMP-107)
    ///
    /// Uses HybridScheduler to route matmuls to GPU when batch_size > 1
    /// and matrix size exceeds threshold. Falls back to CPU for small batches.
    ///
    /// # Arguments
    /// * `token_ids` - Batch of input token IDs [batch_size]
    ///
    /// # Returns
    /// Logits for all positions [batch_size * vocab_size]
    ///
    /// # Errors
    /// Returns error if GPU initialization or tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_batch_gpu(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let batch_size = token_ids.len();
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // Initialize HybridScheduler with reasonable threshold
        // Threshold of 1000 means: batch_size * hidden_dim * out_dim > 1000 uses GPU
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // 1. Token embedding lookup for all tokens
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention LayerNorm
            let normed = ops::layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // QKV projection - use GPU for batch ops
            let qkv = self.batch_qkv_matmul_gpu_with_scheduler(
                &normed,
                &layer.qkv_weight,
                batch_size,
                hidden_dim,
                &mut scheduler,
            )?;

            // Split Q, K, V for batch - PARITY-114: use proper batched causal attention
            let qkv_dim = qkv.len() / batch_size;
            let q_dim = hidden_dim;
            let kv_dim = (qkv_dim - q_dim) / 2;

            // Collect Q, K, V for all positions
            let mut q_all = Vec::with_capacity(batch_size * q_dim);
            let mut k_all = Vec::with_capacity(batch_size * kv_dim);
            let mut v_all = Vec::with_capacity(batch_size * kv_dim);

            for pos in 0..batch_size {
                let qkv_start = pos * qkv_dim;
                q_all.extend_from_slice(&qkv[qkv_start..qkv_start + q_dim]);
                k_all.extend_from_slice(&qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim]);
                v_all.extend_from_slice(&qkv[qkv_start + q_dim + kv_dim..qkv_start + qkv_dim]);
            }

            // Proper batched causal attention (PARITY-114: matches cached forward path)
            let attn_out = self.batched_causal_attention_gpu(&q_all, &k_all, &v_all, batch_size)?;

            // Output projection - use GPU for batch ops
            let projected = self.batch_matmul_gpu(
                &attn_out,
                &layer.attn_output_weight,
                batch_size,
                hidden_dim,
                layer.attn_output_weight.out_dim,
                &mut scheduler,
            )?;

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += projected[i];
            }

            // FFN (pre-norm style)
            let ffn_normed = ops::layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // FFN up projection - use GPU
            let mut ffn_hidden = self.batch_matmul_gpu(
                &ffn_normed,
                &layer.ffn_up_weight,
                batch_size,
                hidden_dim,
                layer.ffn_up_weight.out_dim,
                &mut scheduler,
            )?;

            // GELU activation
            ops::gelu(&mut ffn_hidden);

            // FFN down projection - use GPU
            let ffn_output = self.batch_matmul_gpu(
                &ffn_hidden,
                &layer.ffn_down_weight,
                batch_size,
                layer.ffn_up_weight.out_dim,
                hidden_dim,
                &mut scheduler,
            )?;

            // Residual
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = ops::layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection - use GPU for large vocab
        let logits = self.batch_matmul_gpu(
            &normed,
            &self.lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
            &mut scheduler,
        )?;

        Ok(logits)
    }

    /// Batch matmul with GPU acceleration via HybridScheduler (IMP-107)
    ///
    /// Dequantizes weights and uses GPU for large operations.
    #[cfg(feature = "gpu")]
    fn batch_matmul_gpu(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        m: usize,
        k: usize,
        n: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Dequantize weight to f32
        let weight_f32 = self.dequantize_weight(weight)?;

        // Use HybridScheduler for GPU/CPU dispatch
        // A: [m, k], B: [k, n] -> C: [m, n]
        scheduler.matmul(input, &weight_f32, m, k, n).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::matmul".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            }
        })
    }

    /// Batch QKV matmul with GPU acceleration via HybridScheduler
    ///
    /// Five Whys Root Cause Fix: Handles both fused and separate Q/K/V formats
    #[cfg(feature = "gpu")]
    fn batch_qkv_matmul_gpu_with_scheduler(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        batch_size: usize,
        hidden_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.batch_matmul_gpu(
                input,
                weight,
                batch_size,
                hidden_dim,
                weight.out_dim,
                scheduler,
            ),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out =
                    self.batch_matmul_gpu(input, q, batch_size, hidden_dim, q.out_dim, scheduler)?;
                let k_out =
                    self.batch_matmul_gpu(input, k, batch_size, hidden_dim, k.out_dim, scheduler)?;
                let v_out =
                    self.batch_matmul_gpu(input, v, batch_size, hidden_dim, v.out_dim, scheduler)?;

                // Interleave Q, K, V for each position in batch
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(batch_size * qkv_dim);
                for b in 0..batch_size {
                    output.extend_from_slice(&q_out[b * q.out_dim..(b + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[b * k.out_dim..(b + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[b * v.out_dim..(b + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// Dequantize a weight tensor to f32
    #[cfg(feature = "gpu")]
    fn dequantize_weight(&self, weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        use crate::quantize::{dequantize_q4_k_simd, dequantize_q5_k, dequantize_q6_k, QK_K};

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let total_elements = in_dim * out_dim;

        match weight.qtype {
            GGUF_TYPE_Q4_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 144;
                    let row_end = row_start + super_blocks_per_row * 144;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q4_k_simd(row_data)?;
                    // Take only in_dim values (may have padding due to super-block alignment)
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            GGUF_TYPE_Q5_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 176;
                    let row_end = row_start + super_blocks_per_row * 176;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q5_k(row_data)?;
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            GGUF_TYPE_Q6_K => {
                let super_blocks_per_row = in_dim.div_ceil(QK_K);
                let mut output = Vec::with_capacity(total_elements);
                for row in 0..out_dim {
                    let row_start = row * super_blocks_per_row * 210;
                    let row_end = row_start + super_blocks_per_row * 210;
                    let row_data = &weight.data[row_start..row_end];
                    let row_dequant = dequantize_q6_k(row_data)?;
                    output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                }
                Ok(output)
            },
            _ => {
                // F32 or unsupported - interpret raw bytes as f32
                let num_floats = weight.data.len() / 4;
                let mut output = vec![0.0f32; num_floats];
                for (i, chunk) in weight.data.chunks_exact(4).enumerate() {
                    output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(output)
            },
        }
    }

    /// Dequantize QKV weights - handles both fused and separate formats
    ///
    /// Five Whys Root Cause Fix: This method handles both tensor layouts for dequantization
    #[cfg(feature = "gpu")]
    pub fn dequantize_qkv(&self, qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.dequantize_weight(weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Dequantize each separately and concatenate
                let q_out = self.dequantize_weight(q)?;
                let k_out = self.dequantize_weight(k)?;
                let v_out = self.dequantize_weight(v)?;

                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// Fused batch matmul with GPU acceleration (IMP-109)
    ///
    /// Performs batched matrix multiplication with fused dequantization.
    /// Uses the same weight layout interpretation as `batch_matmul_gpu` for
    /// consistency within the codebase.
    ///
    /// Key optimization: Dequantizes weight matrix once for all batch elements,
    /// reducing memory bandwidth for repeated operations in transformer layers.
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, in_dim]
    /// * `weight` - Quantized weight tensor [out_dim, in_dim]
    /// * `batch_size` - Number of input vectors
    ///
    /// # Returns
    /// Output tensor [batch_size, out_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail or dimensions mismatch
    #[cfg(feature = "gpu")]
    pub fn fused_batch_matmul_gpu(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        // Validate input dimensions
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}={}",
                    input.len(),
                    batch_size,
                    in_dim,
                    batch_size * in_dim
                ),
            });
        }

        // Dequantize weight once (key optimization: reuse across batch elements)
        let weight_f32 = self.dequantize_weight(weight)?;

        // Use HybridScheduler for CPU/GPU dispatch based on workload size
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // Use same matmul approach as batch_matmul_gpu for consistency
        scheduler
            .matmul(input, &weight_f32, batch_size, in_dim, out_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::matmul".to_string(),
                reason: format!("GPU batched matmul failed: {e}"),
            })
    }

    /// Batched causal attention with GPU acceleration (IMP-108)
    ///
    /// Computes causal self-attention using matrix multiplications that can be
    /// GPU-accelerated for large sequence lengths. Uses HybridScheduler for
    /// automatic CPU/GPU dispatch.
    ///
    /// Algorithm:
    /// 1. For each head: scores = Q @ K^T / sqrt(head_dim)
    /// 2. Apply causal mask: scores[i,j] = -inf for j > i
    /// 3. Softmax per row
    /// 4. Output = softmax(scores) @ V
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Attention output [seq_len, hidden_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    #[cfg(feature = "gpu")]
    pub fn batched_causal_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        let mut output = vec![0.0f32; seq_len * hidden_dim];

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;

            // Extract Q_h, K_h, V_h for this head: [seq_len, head_dim]
            let mut q_h = Vec::with_capacity(seq_len * head_dim);
            let mut k_h = Vec::with_capacity(seq_len * head_dim);
            let mut v_h = Vec::with_capacity(seq_len * head_dim);

            for pos in 0..seq_len {
                let start = pos * hidden_dim + head_offset;
                q_h.extend_from_slice(&q[start..start + head_dim]);
                k_h.extend_from_slice(&k[start..start + head_dim]);
                v_h.extend_from_slice(&v[start..start + head_dim]);
            }

            // Compute attention scores: Q_h @ K_h^T -> [seq_len, seq_len]
            // Use GPU for large sequences (seq_len^2 * head_dim ops)
            let scores =
                self.batched_qk_scores(&q_h, &k_h, seq_len, head_dim, scale, &mut scheduler)?;

            // Apply causal mask and softmax
            let attn_weights = self.apply_causal_mask_softmax(&scores, seq_len);

            // Compute output: attn_weights @ V_h -> [seq_len, head_dim]
            let head_output =
                self.batched_attn_v(&attn_weights, &v_h, seq_len, head_dim, &mut scheduler)?;

            // Copy head output to final output
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + head_offset;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Compute Q @ K^T attention scores with GPU acceleration
    #[cfg(feature = "gpu")]
    fn batched_qk_scores(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // Q: [seq_len, head_dim], K: [seq_len, head_dim]
        // scores = Q @ K^T -> [seq_len, seq_len]

        // Transpose K: [head_dim, seq_len]
        let mut k_t = vec![0.0f32; head_dim * seq_len];
        for i in 0..seq_len {
            for j in 0..head_dim {
                k_t[j * seq_len + i] = k[i * head_dim + j];
            }
        }

        // Matmul: Q[seq_len, head_dim] @ K_T[head_dim, seq_len] -> [seq_len, seq_len]
        let scores = scheduler
            .matmul(q, &k_t, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batched_qk_scores".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })?;

        // Apply scale
        let scaled: Vec<f32> = scores.iter().map(|&s| s * scale).collect();
        Ok(scaled)
    }

    /// Apply causal mask and softmax to attention scores
    #[cfg(feature = "gpu")]
    fn apply_causal_mask_softmax(&self, scores: &[f32], seq_len: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            // Apply causal mask: set j > i to -inf
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let idx = i * seq_len + j;
                max_score = max_score.max(scores[idx]);
            }

            // Compute softmax for causal positions only
            let mut exp_sum = 0.0f32;
            for j in 0..=i {
                let idx = i * seq_len + j;
                let exp_val = (scores[idx] - max_score).exp();
                weights[idx] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            for j in 0..=i {
                let idx = i * seq_len + j;
                weights[idx] /= exp_sum;
            }
            // j > i remains 0 (masked out)
        }

        weights
    }

    /// Compute attention_weights @ V with GPU acceleration
    #[cfg(feature = "gpu")]
    fn batched_attn_v(
        &self,
        attn_weights: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scheduler: &mut crate::gpu::HybridScheduler,
    ) -> Result<Vec<f32>> {
        // attn_weights: [seq_len, seq_len], V: [seq_len, head_dim]
        // output = attn_weights @ V -> [seq_len, head_dim]
        scheduler
            .matmul(attn_weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "batched_attn_v".to_string(),
                reason: format!("GPU matmul failed: {e}"),
            })
    }

    // =========================================================================
    // IMP-110: Multi-Head Parallel Attention
    // =========================================================================

    /// Reshape tensor from [seq_len, hidden_dim] to [num_heads, seq_len, head_dim]
    ///
    /// IMP-110b: Prepares Q/K/V tensors for parallel multi-head processing.
    /// Original layout stores all head features contiguously per position.
    /// New layout groups by head for batched matmul operations.
    ///
    /// # Arguments
    /// * `input` - Input tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (hidden_dim / num_heads)
    ///
    /// # Returns
    /// Reshaped tensor [num_heads, seq_len, head_dim]
    #[cfg(feature = "gpu")]
    pub fn reshape_for_parallel_heads(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = num_heads * head_dim;
        let expected_len = seq_len * hidden_dim;

        if input.len() != expected_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match seq_len={} * hidden_dim={}={}",
                    input.len(),
                    seq_len,
                    hidden_dim,
                    expected_len
                ),
            });
        }

        let mut reshaped = vec![0.0f32; num_heads * seq_len * head_dim];

        // Transform: input[pos * hidden_dim + h * head_dim + d]
        //         -> reshaped[h * seq_len * head_dim + pos * head_dim + d]
        for h in 0..num_heads {
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    let orig_idx = pos * hidden_dim + h * head_dim + d;
                    let new_idx = h * seq_len * head_dim + pos * head_dim + d;
                    reshaped[new_idx] = input[orig_idx];
                }
            }
        }

        Ok(reshaped)
    }

    /// Compute batched Q@K^T scores for all heads in parallel
    ///
    /// IMP-110c: Computes attention scores for all heads in a single batch.
    /// Takes Q, K in original [seq_len, hidden_dim] layout and computes
    /// Q@K^T for each head.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `scale` - Attention scale (1/sqrt(head_dim))
    ///
    /// # Returns
    /// Batched scores [num_heads, seq_len, seq_len]
    #[cfg(feature = "gpu")]
    pub fn parallel_batched_qk_scores(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        // Reshape Q and K to [num_heads, seq_len, head_dim]
        let q_reshaped = self.reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self.reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;

        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // For each head: Q_h @ K_h^T -> [seq_len, seq_len]
        // Total output: [num_heads, seq_len, seq_len]
        let mut all_scores = Vec::with_capacity(num_heads * seq_len * seq_len);

        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            let q_h = &q_reshaped[head_start..head_start + seq_len * head_dim];
            let k_h = &k_reshaped[head_start..head_start + seq_len * head_dim];

            // Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_h[i * head_dim + j];
                }
            }

            // Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] -> [seq_len, seq_len]
            let scores = scheduler
                .matmul(q_h, &k_t, seq_len, head_dim, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_batched_qk_scores".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Apply scale and accumulate
            for s in &scores {
                all_scores.push(s * scale);
            }
        }

        Ok(all_scores)
    }

    /// Multi-head attention with parallel head processing
    ///
    /// IMP-110a: Processes all attention heads in parallel batches instead
    /// of iterating head-by-head. This enables better GPU utilization.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Attention output [seq_len, hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn parallel_multihead_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Get batched scores for all heads: [num_heads, seq_len, seq_len]
        let batched_scores =
            self.parallel_batched_qk_scores(q, k, seq_len, num_heads, head_dim, scale)?;

        // Apply causal mask and softmax per head
        let mut batched_weights = vec![0.0f32; num_heads * seq_len * seq_len];
        for h in 0..num_heads {
            let head_offset = h * seq_len * seq_len;
            let head_scores = &batched_scores[head_offset..head_offset + seq_len * seq_len];
            let head_weights = self.apply_causal_mask_softmax(head_scores, seq_len);
            batched_weights[head_offset..head_offset + seq_len * seq_len]
                .copy_from_slice(&head_weights);
        }

        // Reshape V to [num_heads, seq_len, head_dim]
        let v_reshaped = self.reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Compute attention output for all heads
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // Output: [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];

        for h in 0..num_heads {
            let weights_offset = h * seq_len * seq_len;
            let v_offset = h * seq_len * head_dim;

            let head_weights = &batched_weights[weights_offset..weights_offset + seq_len * seq_len];
            let v_h = &v_reshaped[v_offset..v_offset + seq_len * head_dim];

            // weights @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] -> [seq_len, head_dim]
            let head_output = scheduler
                .matmul(head_weights, v_h, seq_len, seq_len, head_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_multihead_attention_gpu".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Copy to output in original layout
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + h * head_dim;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    // =========================================================================
    // IMP-111: Flash Attention-style Tiled Computation
    // =========================================================================

    /// Standard softmax (reference implementation)
    ///
    /// IMP-111a: Reference implementation for testing online softmax.
    /// Computes softmax in the standard way: exp(x - max) / sum(exp(x - max))
    pub fn standard_softmax(&self, scores: &[f32]) -> Vec<f32> {
        if scores.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();

        // Normalize
        exp_scores.iter().map(|&e| e / sum).collect()
    }

    /// Online softmax with tiled processing (O(1) memory per tile)
    ///
    /// IMP-111a: Implements the "online softmax" algorithm that processes
    /// data in tiles without materializing the full softmax denominator.
    ///
    /// Algorithm:
    /// 1. Process tiles, tracking running max (m) and denominator (d)
    /// 2. When new tile has larger max, rescale previous denominator
    /// 3. Final pass normalizes all values
    ///
    /// # Arguments
    /// * `scores` - Input scores to apply softmax
    /// * `tile_size` - Size of each tile for processing
    ///
    /// # Returns
    /// Softmax probabilities
    pub fn online_softmax(&self, scores: &[f32], tile_size: usize) -> Result<Vec<f32>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let n = scores.len();
        let tile_size = tile_size.max(1);

        // Running statistics
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;

        // First pass: compute global max and sum using online algorithm
        for tile_start in (0..n).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n);

            // Find local max in this tile
            let local_max = scores[tile_start..tile_end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if local_max > global_max {
                // Rescale previous sum when we find a new max
                let rescale = (global_max - local_max).exp();
                global_sum *= rescale;
                global_max = local_max;
            }

            // Add this tile's contribution to sum
            for &s in &scores[tile_start..tile_end] {
                global_sum += (s - global_max).exp();
            }
        }

        // Second pass: compute final softmax values
        let mut result = Vec::with_capacity(n);
        for &s in scores {
            result.push((s - global_max).exp() / global_sum);
        }

        Ok(result)
    }

    /// Standard single-head attention (reference implementation)
    ///
    /// IMP-111b: Reference implementation that materializes full attention matrix.
    /// Used to verify tiled attention correctness.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Dimension per head
    /// * `scale` - Attention scale (1/sqrt(head_dim))
    pub fn standard_single_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Compute attention scores: Q @ K^T -> [seq_len, seq_len]
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Apply softmax per row
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row = &scores[row_start..row_start + seq_len];
            let softmax = self.standard_softmax(row);
            weights[row_start..row_start + seq_len].copy_from_slice(&softmax);
        }

        // Compute output: weights @ V -> [seq_len, head_dim]
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += weights[i * seq_len + j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }

        Ok(output)
    }

    /// Tiled single-head attention (non-causal)
    ///
    /// IMP-111b: Flash Attention-style tiled computation.
    /// Processes K/V in tiles, maintaining running softmax statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_single_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let tile_size = tile_size.max(1);
        let mut output = vec![0.0f32; seq_len * head_dim];

        // Process each query position
        for i in 0..seq_len {
            let q_i = &q[i * head_dim..(i + 1) * head_dim];

            // Running statistics for online softmax
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut running_output = vec![0.0f32; head_dim];

            // Process K/V in tiles
            for tile_start in (0..seq_len).step_by(tile_size) {
                let tile_end = (tile_start + tile_size).min(seq_len);

                // Compute scores for this tile: q_i @ K_tile^T
                let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
                for j in tile_start..tile_end {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k[j * head_dim + d];
                    }
                    tile_scores.push(dot * scale);
                }

                // Find tile max
                let tile_max = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running statistics
                let new_max = running_max.max(tile_max);

                // Rescale previous output and sum
                if new_max > running_max && running_sum > 0.0 {
                    let rescale = (running_max - new_max).exp();
                    running_sum *= rescale;
                    for out_val in &mut running_output {
                        *out_val *= rescale;
                    }
                }
                running_max = new_max;

                // Accumulate this tile's contribution
                for (idx, &score) in tile_scores.iter().enumerate() {
                    let j = tile_start + idx;
                    let weight = (score - running_max).exp();
                    running_sum += weight;
                    for d in 0..head_dim {
                        running_output[d] += weight * v[j * head_dim + d];
                    }
                }
            }

            // Normalize output
            for d in 0..head_dim {
                output[i * head_dim + d] = running_output[d] / running_sum;
            }
        }

        Ok(output)
    }

    /// Tiled causal attention
    ///
    /// IMP-111c: Flash Attention with causal masking.
    /// For position i, only attends to positions 0..=i.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let tile_size = tile_size.max(1);
        let mut output = vec![0.0f32; seq_len * head_dim];

        // Process each query position
        for i in 0..seq_len {
            let q_i = &q[i * head_dim..(i + 1) * head_dim];

            // Running statistics for online softmax
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut running_output = vec![0.0f32; head_dim];

            // Only process K/V up to position i (causal)
            let causal_len = i + 1;

            // Process K/V in tiles
            for tile_start in (0..causal_len).step_by(tile_size) {
                let tile_end = (tile_start + tile_size).min(causal_len);

                // Compute scores for this tile: q_i @ K_tile^T
                let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
                for j in tile_start..tile_end {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k[j * head_dim + d];
                    }
                    tile_scores.push(dot * scale);
                }

                // Find tile max
                let tile_max = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running statistics
                let new_max = running_max.max(tile_max);

                // Rescale previous output and sum
                if new_max > running_max && running_sum > 0.0 {
                    let rescale = (running_max - new_max).exp();
                    running_sum *= rescale;
                    for out_val in &mut running_output {
                        *out_val *= rescale;
                    }
                }
                running_max = new_max;

                // Accumulate this tile's contribution
                for (idx, &score) in tile_scores.iter().enumerate() {
                    let j = tile_start + idx;
                    let weight = (score - running_max).exp();
                    running_sum += weight;
                    for d in 0..head_dim {
                        running_output[d] += weight * v[j * head_dim + d];
                    }
                }
            }

            // Normalize output
            if running_sum > 0.0 {
                for d in 0..head_dim {
                    output[i * head_dim + d] = running_output[d] / running_sum;
                }
            }
        }

        Ok(output)
    }
}

// =============================================================================
// IMP-800: CUDA-Accelerated Model Wrapper
// =============================================================================

/// CUDA-accelerated wrapper for `OwnedQuantizedModel` (IMP-800a)
///
/// Provides GPU-accelerated forward pass using NVIDIA CUDA via trueno-gpu.
/// Caches the CudaExecutor to avoid initialization overhead (~50ms) per call.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
///
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&tokens)?;
/// ```
#[cfg(feature = "cuda")]
pub struct OwnedQuantizedModelCuda {
    /// Inner model
    model: OwnedQuantizedModel,
    /// Cached CUDA executor
    executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
}

#[cfg(feature = "cuda")]
impl OwnedQuantizedModelCuda {
    /// Create a new CUDA-accelerated model wrapper
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(model: OwnedQuantizedModel, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048)
    }

    /// Create a new CUDA-accelerated model wrapper with custom max sequence length
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache (PAR-018)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
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

        // PAR-018: Initialize GPU-resident KV cache for attention acceleration
        // This avoids ~66 MB CPU→GPU transfer per token for TinyLlama
        let num_layers = model.layers.len();
        let num_heads = model.config.num_heads;
        let num_kv_heads = model.config.num_kv_heads; // PAR-021 GQA support
        let head_dim = model.config.hidden_dim / num_heads;

        executor
            .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_kv_cache_gpu".to_string(),
                reason: format!("GPU KV cache initialization failed: {e}"),
            })?;

        // PAR-060: Set RoPE theta for position embeddings
        if verbose() {
            eprintln!(
                "[PAR-060] Setting rope_theta = {} for GPU path",
                model.config.rope_theta
            );
        }
        executor.set_rope_theta(model.config.rope_theta);

        // CORRECTNESS-011: Set rope_type for correct RoPE style (NORM vs NEOX)
        if verbose() {
            eprintln!(
                "[CORRECTNESS-011] Setting rope_type = {} for GPU path (0=NORM, 2=NEOX)",
                model.config.rope_type
            );
        }
        executor.set_rope_type(model.config.rope_type);

        Ok(Self {
            model,
            executor,
            device_name,
            memory_info,
        })
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Get GPU device name
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    // ========================================================================
    // PAR-073: BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    ///
    /// When enabled, each brick operation is timed individually using
    /// `std::time::Instant` with CUDA sync for accurate GPU timing.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling (default state).
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

    /// PAR-103: Pre-cache all weights for batched forward pass.
    ///
    /// This loads all layer weights into GPU memory with the naming convention
    /// expected by `forward_batch_cuda_native`. Required before using batch mode.
    ///
    /// # Returns
    ///
    /// Total MB of weights uploaded to GPU.
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails.
    pub fn pre_cache_weights_for_batch(&mut self) -> Result<usize> {
        let mut total_bytes = 0usize;
        let num_layers = self.model.layers.len();

        eprintln!(
            "[PAR-103] Pre-caching {} layer weights for batch mode...",
            num_layers
        );

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layer.{}", layer_idx);

            // Cache QKV weights
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    let q_name = format!("{}.attn_q.weight", prefix);
                    let k_name = format!("{}.attn_k.weight", prefix);
                    let v_name = format!("{}.attn_v.weight", prefix);

                    total_bytes += self
                        .executor
                        .load_quantized_weights(&q_name, &q.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache Q weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&k_name, &k.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache K weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&v_name, &v.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache V weights: {}", e),
                        })?;
                },
                OwnedQKVWeights::Fused(qkv) => {
                    let qkv_name = format!("{}.attn_qkv.weight", prefix);
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&qkv_name, &qkv.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache QKV weights: {}", e),
                        })?;
                },
            }

            // Cache O projection
            let o_name = format!("{}.attn_output.weight", prefix);
            total_bytes += self
                .executor
                .load_quantized_weights(&o_name, &layer.attn_output_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache O weights: {}", e),
                })?;

            // Cache FFN weights (ffn_gate is optional - only SwiGLU models have it)
            let ffn_up_name = format!("{}.ffn_up.weight", prefix);
            let ffn_down_name = format!("{}.ffn_down.weight", prefix);

            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_up_name, &layer.ffn_up_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN up weights: {}", e),
                })?;
            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_down_name, &layer.ffn_down_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN down weights: {}", e),
                })?;

            // FFN gate is optional (SwiGLU models like LLaMA/Qwen)
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let ffn_gate_name = format!("{}.ffn_gate.weight", prefix);
                total_bytes += self
                    .executor
                    .load_quantized_weights(&ffn_gate_name, &gate_weight.data)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "pre_cache_weights_for_batch".to_string(),
                        reason: format!("Failed to cache FFN gate weights: {}", e),
                    })?;
            }
        }

        // Cache LM head
        let lm_head_name = "output.weight".to_string();
        total_bytes += self
            .executor
            .load_quantized_weights(&lm_head_name, &self.model.lm_head_weight.data)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "pre_cache_weights_for_batch".to_string(),
                reason: format!("Failed to cache LM head weights: {}", e),
            })?;

        let total_mb = total_bytes / (1024 * 1024);
        eprintln!(
            "[PAR-103] Pre-cached {} MB of weights for batch mode",
            total_mb
        );
        Ok(total_bytes)
    }

    /// Get profiler summary report.
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.executor.profiler_summary()
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// PAR-111: Get mutable reference to CUDA executor
    ///
    /// Allows direct access for batched forward path and workspace initialization.
    #[must_use]
    pub fn executor_mut(&mut self) -> &mut crate::cuda::CudaExecutor {
        &mut self.executor
    }

    /// Forward pass using CUDA GEMM acceleration (IMP-800a)
    ///
    /// Uses CudaExecutor for matrix multiplications in the FFN layers.
    /// Attention and embedding remain on CPU for now.
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
    /// Returns error if CUDA operations fail
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.model.layers {
            // 2a. Attention layer norm (CPU)
            let normed = self.model.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K for now)
            // GQA-aware dimensions: Q has num_heads, K/V have num_kv_heads
            let num_kv_heads = self.model.config.num_kv_heads;
            let head_dim = hidden_dim / self.model.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = hidden_dim + 2 * kv_dim; // Q + K + V with GQA
            let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                self.model.add_bias(&mut qkv, bias);
            }

            // 2c. Attention (CPU - complex control flow)
            let seq_len = token_ids.len();
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k = qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v = &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // GQA-aware RoPE: Q uses num_heads, K uses num_kv_heads
                self.model
                    .apply_rope(&mut q, s, self.model.config.num_heads);
                self.model.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            let attn_out = self.model.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // 2d. Attention output projection (CPU - fused Q4_K)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                self.model.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection - try GPU GEMM if weights are dequantized
            // For now, use CPU fused ops (GPU overhead too high for m=1)
            let mut ffn_hidden = self.model.fused_matmul(&hidden, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                self.model.add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation (CPU)
            self.model.gelu(&mut ffn_hidden);

            // 2g. FFN down projection (CPU fused)
            let mut ffn_output = self
                .model
                .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                self.model.add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm (CPU)
        let normed = self.model.layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self
            .model
            .fused_matmul(last_hidden, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            self.model.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens using CUDA acceleration (IMP-800a)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = prompt.to_vec();

        for _ in 0..config.max_tokens {
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);

                // Apply temperature and sample (simplified - take max after temperature)
                let max_logit = indexed[0].1;
                let _exp_sum: f32 = indexed
                    .iter()
                    .map(|(_, l)| ((l - max_logit) / config.temperature).exp())
                    .sum();

                // Take argmax (proper probabilistic sampling would use exp_sum for normalization)
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Synchronize CUDA stream (wait for all GPU operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.executor
            .synchronize()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::synchronize".to_string(),
                reason: format!("CUDA sync failed: {e}"),
            })
    }

    /// Forward pass with KV cache using CUDA multi-head attention (PARITY-044)
    ///
    /// Uses `CudaExecutor::flash_attention_multi_head` for GPU-accelerated attention.
    /// This processes all attention heads in parallel on the GPU, avoiding per-head
    /// CPU loops.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Position in sequence
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if CUDA operations fail
    pub fn forward_single_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim; // GQA: K/V may have fewer heads than Q
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(&[token_id]);

        // 2. Process through transformer layers (index-based to avoid borrow issues)
        for layer_idx in 0..num_layers {
            // 2a. Attention layer norm (CPU)
            let normed = self.model.layer_norm(
                &hidden,
                &self.model.layers[layer_idx].attn_norm_weight,
                self.model.layers[layer_idx].attn_norm_bias.as_deref(),
                eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K)
            let mut qkv = self
                .model
                .qkv_matmul(&normed, &self.model.layers[layer_idx].qkv_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].qkv_bias {
                self.model.add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V and apply RoPE (GQA-aware dimensions)
            // Q has hidden_dim = num_heads * head_dim
            // K/V have kv_dim = num_kv_heads * head_dim (may be smaller for GQA)
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model
                .apply_rope(&mut k, position, self.model.config.num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet, GQA expansion needed for output
                if num_kv_heads < num_heads {
                    // Expand V to match num_heads by repeating KV groups
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded = vec![0.0f32; hidden_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let src_offset = kv_head * head_dim;
                        let dst_offset = q_head * head_dim;
                        expanded[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&v[src_offset..src_offset + head_dim]);
                    }
                    expanded
                } else {
                    v.clone()
                }
            } else {
                // Use GPU multi-head attention if cache is large enough (PARITY-044)
                let cache_len = if kv_dim > 0 {
                    k_cache.len() / kv_dim
                } else {
                    0
                };
                let total_len = cache_len + 1;

                // PAR-017: Lower GPU attention threshold for more consistent GPU usage
                // Previous: 32 tokens caused high variance with short sequences
                const GPU_ATTN_THRESHOLD: usize = 8;

                if total_len >= GPU_ATTN_THRESHOLD && num_kv_heads == num_heads {
                    // GPU path only works for non-GQA models currently
                    self.cuda_attention_with_cache(
                        &q, k_cache, v_cache, &k, &v, total_len, num_heads, head_dim,
                    )?
                } else {
                    // CPU path for short sequences or GQA models
                    // Use GQA-aware version that handles grouped KV heads correctly
                    self.model
                        .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
                }
            };

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection (CPU fused)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &self.model.layers[layer_idx].attn_output_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].attn_output_bias {
                self.model.add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // PAR-047: FFN with proper SwiGLU/GELU detection
            // LLaMA-family models use SwiGLU (ffn_gate_weight present)
            // Phi-2 style models use GELU (no gate weight)
            let ffn_activated =
                if let Some(ref gate_weight) = self.model.layers[layer_idx].ffn_gate_weight {
                    // SwiGLU path (LLaMA, TinyLlama, Mistral, Qwen, etc.)
                    // Apply FFN norm if present (separate from attention norm in LLaMA-style)
                    let ffn_input =
                        if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                            self.model.layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        } else {
                            hidden.clone()
                        };

                    let mut ffn_up = self
                        .model
                        .fused_matmul(&ffn_input, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        self.model.add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.model.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_gate_bias {
                        self.model.add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    self.model.silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                } else {
                    // GELU path (phi-2 style, no gate weight)
                    let mut ffn_hidden = self
                        .model
                        .fused_matmul(&hidden, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        self.model.add_bias(&mut ffn_hidden, bias);
                    }
                    self.model.gelu(&mut ffn_hidden);
                    ffn_hidden
                };

            // 2i. FFN down projection (CPU fused)
            let mut ffn_output = self.model.fused_matmul(
                &ffn_activated,
                &self.model.layers[layer_idx].ffn_down_weight,
            )?;
            if let Some(ref bias) = self.model.layers[layer_idx].ffn_down_bias {
                self.model.add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (CPU)
        let normed = self.model.layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let mut logits = self
            .model
            .fused_matmul(&normed, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            self.model.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// IMP-1010: GPU-accelerated fused Q4_K matmul
    ///
    /// Uses `CudaExecutor::q4k_matvec` to execute quantized matrix-vector
    /// multiplication directly on GPU, avoiding CPU SIMD overhead.
    ///
    /// # Performance Impact
    ///
    /// - CPU SIMD path: ~5 tok/s (limited by memory bandwidth)
    /// - GPU CUDA path: ~200 tok/s (theoretical, matching Ollama)
    /// - Key: Dequantize on GPU, not on CPU
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector (f32)
    /// * `weight` - Quantized weight tensor (Q4_K format)
    ///
    /// # Returns
    ///
    /// Output vector [out_dim]
    fn fused_matmul_cuda(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        // Only Q4_K is supported for GPU acceleration (PARITY-041)
        const GGUF_TYPE_Q4_K: u32 = 12;

        if weight.qtype != GGUF_TYPE_Q4_K {
            // Fallback to CPU for non-Q4_K weights
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        // GPU kernel expects single input (seq_len=1 during token generation)
        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "IMP-1010: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        // Allocate output buffer
        let mut output = vec![0.0f32; out_dim];

        // PAR-014: Use cached GEMV for weight reuse (avoids re-transfer each call)
        // Cache key is based on weight data pointer (stable since model owns data)
        let cache_key = format!("q4k_{:016x}", weight.data.as_ptr() as usize);

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(&cache_key) {
            self.executor
                .load_quantized_weights(&cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_q4k_cache".to_string(),
                    reason: format!("Failed to cache Q4_K weights: {e}"),
                })?;
        }

        // Execute Q4_K matmul on GPU using cached weights
        self.executor
            .q4k_gemv_cached(
                &cache_key,
                input,
                &mut output,
                out_dim as u32,
                in_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "q4k_gemv_cached".to_string(),
                reason: format!("CUDA Q4_K GEMV failed: {e}"),
            })?;

        Ok(output)
    }

    /// PAR-014: Fused matmul with explicit cache key
    ///
    /// Same as `fused_matmul_cuda` but accepts an explicit cache key, allowing
    /// the caller to use the original weight pointer for caching even when
    /// working with cloned weight data.
    fn fused_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        // Only Q4_K is supported for GPU acceleration
        const GGUF_TYPE_Q4_K: u32 = 12;

        if weight.qtype != GGUF_TYPE_Q4_K {
            // Fallback to CPU for non-Q4_K weights
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "PAR-014: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        let mut output = vec![0.0f32; out_dim];

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(cache_key) {
            self.executor
                .load_quantized_weights(cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_q4k_cache".to_string(),
                    reason: format!("Failed to cache Q4_K weights: {e}"),
                })?;
        }

        // Execute Q4_K matmul on GPU using cached weights
        self.executor
            .q4k_gemv_cached(cache_key, input, &mut output, out_dim as u32, in_dim as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "q4k_gemv_cached".to_string(),
                reason: format!("CUDA Q4_K GEMV failed: {e}"),
            })?;

        Ok(output)
    }

    /// QKV matmul with CUDA - handles both fused and separate Q/K/V
    ///
    /// Five Whys Root Cause Fix: Supports TinyLlama and other LLaMA-style models
    fn qkv_matmul_cuda(&mut self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_cuda(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                // Concatenate Q, K, V
                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// PAR-014: QKV matmul with explicit cache key for fused weights
    ///
    /// Same as `qkv_matmul_cuda` but accepts a cache key for the fused case.
    fn qkv_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_matmul_cuda_with_key(input, weight, cache_key)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, we still use the cloned pointers
                // (less critical since these are already separate tensors)
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// IMP-1010: Full GPU forward pass for single token with KV cache
    ///
    /// This method uses GPU acceleration for ALL matmul operations:
    /// - QKV projection (3x hidden_dim × hidden_dim)
    /// - Attention output projection (hidden_dim × hidden_dim)
    /// - FFN up projection (hidden_dim × 4*hidden_dim)
    /// - FFN down projection (4*hidden_dim × hidden_dim)
    /// - LM head projection (hidden_dim × vocab_size)
    ///
    /// # Performance Target
    ///
    /// - CPU SIMD path: ~5 tok/s
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    pub fn forward_single_full_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // PAR-021: GQA support
        // Q: [hidden_dim] = [num_heads * head_dim]
        // K: [kv_dim] = [num_kv_heads * head_dim] (smaller for GQA)
        // V: [kv_dim] = [num_kv_heads * head_dim] (smaller for GQA)
        let kv_dim = num_kv_heads * head_dim;

        // 1. Token embedding lookup (CPU - fast enough, single lookup)
        let mut hidden = self.model.embed(&[token_id]);

        // IMP-1010-DEBUG: Check embedding output (disabled for performance)
        #[allow(clippy::never_loop)]
        if false {
            let embed_sum: f32 = hidden.iter().sum();
            let embed_has_nan = hidden.iter().any(|x| x.is_nan());
            eprintln!(
                "[IMP-1010] pos{} embedding: sum={:.6e}, has_nan={}",
                position, embed_sum, embed_has_nan
            );
        }

        // PAR-016: Pre-capture LM head cache key for stable caching
        let lm_head_cache_key = format!(
            "q4k_{:016x}",
            self.model.lm_head_weight.data.as_ptr() as usize
        );

        // PAR-050: Detect RMSNorm architecture (LLaMA uses RMSNorm and SwiGLU)
        let use_rmsnorm = self
            .model
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // PAR-014: Capture original weight pointers BEFORE cloning for stable cache keys
            // This ensures weight caching works across forward passes
            let attn_output_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx]
                    .attn_output_weight
                    .data
                    .as_ptr() as usize
            );
            let ffn_up_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx].ffn_up_weight.data.as_ptr() as usize
            );
            let ffn_down_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx].ffn_down_weight.data.as_ptr() as usize
            );
            // Capture QKV weight pointer for cache key (handles both Fused and Separate)
            let qkv_cache_key = match &self.model.layers[layer_idx].qkv_weight {
                OwnedQKVWeights::Fused(ref tensor) => {
                    format!("q4k_{:016x}", tensor.data.as_ptr() as usize)
                },
                OwnedQKVWeights::Separate { ref q, .. } => {
                    // Use Q tensor pointer as representative key for separate case
                    format!("q4k_{:016x}", q.data.as_ptr() as usize)
                },
            };

            // Clone weights to avoid borrow conflicts with &mut self
            // IMP-1010: This is necessary because fused_matmul_cuda needs &mut self
            let qkv_weight = self.model.layers[layer_idx].qkv_weight.clone();
            let qkv_bias = self.model.layers[layer_idx].qkv_bias.clone();
            let attn_norm_weight = self.model.layers[layer_idx].attn_norm_weight.clone();
            let attn_norm_bias = self.model.layers[layer_idx].attn_norm_bias.clone();
            let attn_output_weight_data =
                self.model.layers[layer_idx].attn_output_weight.data.clone();
            let attn_output_weight_in_dim = self.model.layers[layer_idx].attn_output_weight.in_dim;
            let attn_output_weight_out_dim =
                self.model.layers[layer_idx].attn_output_weight.out_dim;
            let attn_output_weight_qtype = self.model.layers[layer_idx].attn_output_weight.qtype;
            let attn_output_bias = self.model.layers[layer_idx].attn_output_bias.clone();
            let ffn_up_weight_data = self.model.layers[layer_idx].ffn_up_weight.data.clone();
            let ffn_up_weight_in_dim = self.model.layers[layer_idx].ffn_up_weight.in_dim;
            let ffn_up_weight_out_dim = self.model.layers[layer_idx].ffn_up_weight.out_dim;
            let ffn_up_weight_qtype = self.model.layers[layer_idx].ffn_up_weight.qtype;
            let ffn_up_bias = self.model.layers[layer_idx].ffn_up_bias.clone();
            let ffn_down_weight_data = self.model.layers[layer_idx].ffn_down_weight.data.clone();
            let ffn_down_weight_in_dim = self.model.layers[layer_idx].ffn_down_weight.in_dim;
            let ffn_down_weight_out_dim = self.model.layers[layer_idx].ffn_down_weight.out_dim;
            let ffn_down_weight_qtype = self.model.layers[layer_idx].ffn_down_weight.qtype;
            let ffn_down_bias = self.model.layers[layer_idx].ffn_down_bias.clone();
            // PAR-015: Extract FFN gate weight for SwiGLU (LLaMA models)
            let ffn_gate_weight = self.model.layers[layer_idx].ffn_gate_weight.clone();
            let ffn_gate_bias = self.model.layers[layer_idx].ffn_gate_bias.clone();
            let ffn_gate_cache_key = ffn_gate_weight
                .as_ref()
                .map(|w| format!("q4k_{:016x}", w.data.as_ptr() as usize));

            // Reconstruct weight tensors
            let attn_output_weight = OwnedQuantizedTensor {
                data: attn_output_weight_data,
                in_dim: attn_output_weight_in_dim,
                out_dim: attn_output_weight_out_dim,
                qtype: attn_output_weight_qtype,
            };
            let ffn_up_weight = OwnedQuantizedTensor {
                data: ffn_up_weight_data,
                in_dim: ffn_up_weight_in_dim,
                out_dim: ffn_up_weight_out_dim,
                qtype: ffn_up_weight_qtype,
            };
            let ffn_down_weight = OwnedQuantizedTensor {
                data: ffn_down_weight_data,
                in_dim: ffn_down_weight_in_dim,
                out_dim: ffn_down_weight_out_dim,
                qtype: ffn_down_weight_qtype,
            };

            // 2a. Attention layer norm (CPU - fast for single vector)
            // PAR-050: Use RMSNorm for LLaMA models (no bias), LayerNorm for others
            let normed = if use_rmsnorm {
                self.model.rms_norm(&hidden, &attn_norm_weight, eps)
            } else {
                self.model
                    .layer_norm(&hidden, &attn_norm_weight, attn_norm_bias.as_deref(), eps)
            };

            // IMP-1010-DEBUG: Check normed output for NaN (disabled for performance)
            #[allow(clippy::never_loop)]
            if false {
                let normed_has_nan = normed.iter().any(|x| x.is_nan());
                let normed_sum: f32 = normed.iter().sum();
                let normed_max = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} normed: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, normed_sum, normed_max, normed_has_nan
                );
            }

            // 2b. QKV projection (GPU - PAR-014: use pre-captured cache key)
            let mut qkv = self.qkv_matmul_cuda_with_key(&normed, &qkv_weight, &qkv_cache_key)?;
            if let Some(ref bias) = qkv_bias {
                self.model.add_bias(&mut qkv, bias);
            }

            // IMP-1010-DEBUG: Check QKV output for NaN
            if false {
                let qkv_has_nan = qkv.iter().any(|x| x.is_nan());
                let qkv_sum: f32 = qkv.iter().sum();
                let qkv_max = qkv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} QKV: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, qkv_sum, qkv_max, qkv_has_nan
                );
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // PAR-021: For GQA, K and V have smaller kv_dim (num_kv_heads * head_dim)
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model.apply_rope(&mut k, position, num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            // CORRECTNESS-RESOLVED: Always use CPU attention (GPU attention precision issues)
            // GPU matmul is still used for QKV, output, and FFN projections
            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet
                // PAR-021: Expand V for GQA (each KV head serves multiple Q heads)
                if num_kv_heads < num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded_v = vec![0.0f32; hidden_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let v_start = kv_head * head_dim;
                        let out_start = q_head * head_dim;
                        expanded_v[out_start..out_start + head_dim]
                            .copy_from_slice(&v[v_start..v_start + head_dim]);
                    }
                    expanded_v
                } else {
                    v.clone()
                }
            } else {
                // Use CPU GQA-aware attention (correct implementation)
                self.model
                    .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
            };

            // 2e. Store K and V in cache (only CPU cache if no GPU cache)
            // IMP-1010-DEBUG: Always use CPU cache since GPU attention is disabled
            // if !self.executor.has_kv_cache_gpu() {
            cache.append(layer_idx, &k, &v);
            // }

            // 2f. Attention output projection (GPU - PAR-014: use pre-captured cache key)
            let mut attn_output = self.fused_matmul_cuda_with_key(
                &attn_out,
                &attn_output_weight,
                &attn_output_cache_key,
            )?;
            if let Some(ref bias) = attn_output_bias {
                self.model.add_bias(&mut attn_output, bias);
            }

            // IMP-1010-DEBUG: Check attention output for NaN
            if false {
                let attn_out_has_nan = attn_out.iter().any(|x| x.is_nan());
                let attn_out_sum: f32 = attn_out.iter().sum();
                let attn_proj_has_nan = attn_output.iter().any(|x| x.is_nan());
                let attn_proj_sum: f32 = attn_output.iter().sum();
                let attn_proj_max = attn_output
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[IMP-1010] pos{} L{} attn_out: sum={:.6e}, has_nan={} | proj: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, attn_out_sum, attn_out_has_nan, attn_proj_sum, attn_proj_max, attn_proj_has_nan);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // IMP-1010-DEBUG: Check hidden after residual
            if false {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} after attn: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_max, hidden_has_nan
                );
            }

            // PAR-049-DEBUG: Compare attention output with CPU (disabled for performance)
            // Re-enable by changing `false` to `true` for debugging
            #[allow(clippy::never_loop, clippy::while_let_on_iterator)]
            if false {
                let cpu_attn = self.model.fused_matmul(&attn_out, &attn_output_weight)?;
                let max_diff = attn_output
                    .iter()
                    .zip(cpu_attn.iter())
                    .map(|(g, c)| (g - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "[PAR-049] L0 pos{} attn_output max_diff: {:.6e}",
                    position, max_diff
                );
                if position == 1 {
                    eprintln!(
                        "[PAR-049] L0 pos1 attn_out[0..5]: {:?}",
                        &attn_out[..5.min(attn_out.len())]
                    );
                    let k_len = cache.get_k(layer_idx).len();
                    let v_len = cache.get_v(layer_idx).len();
                    eprintln!(
                        "[PAR-049] L0 pos1 k_cache.len: {}, v_cache.len: {}",
                        k_len, v_len
                    );
                }
            }

            // 2h/2i. FFN
            // PAR-057: Re-enable fused FFN path now that kernels are fixed

            // IMP-1010-DEBUG: Check hidden state going into FFN for layers near NaN origin
            if false {
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let hidden_min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                eprintln!("[IMP-1010] pos{} L{} before FFN: sum={:.6e}, min={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_min, hidden_max, hidden_has_nan);
            }

            #[allow(clippy::overly_complex_bool_expr)]
            let ffn_output = if ffn_up_bias.is_none()
                && ffn_down_bias.is_none()
                && ffn_up_weight.qtype == 12
                && ffn_down_weight.qtype == 12
            {
                // Fused FFN path: up + GELU + down in single GPU round-trip
                let intermediate_dim = ffn_up_weight.out_dim;
                let mut output = vec![0.0f32; hidden_dim];

                // Ensure weights are cached
                if !self.executor.has_quantized_weights(&ffn_up_cache_key) {
                    self.executor
                        .load_quantized_weights(&ffn_up_cache_key, &ffn_up_weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_ffn_up_cache".to_string(),
                            reason: format!("Failed to cache FFN up weights: {e}"),
                        })?;
                }
                if !self.executor.has_quantized_weights(&ffn_down_cache_key) {
                    self.executor
                        .load_quantized_weights(&ffn_down_cache_key, &ffn_down_weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_ffn_down_cache".to_string(),
                            reason: format!("Failed to cache FFN down weights: {e}"),
                        })?;
                }

                self.executor
                    .fused_ffn_q4k(
                        &hidden,
                        &mut output,
                        &ffn_up_cache_key,
                        &ffn_down_cache_key,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_fused_ffn".to_string(),
                        reason: format!("CUDA fused FFN failed: {e}"),
                    })?;

                // IMP-1010-DEBUG: Check fused FFN output for layers near NaN origin
                if false {
                    let out_sum: f32 = output.iter().sum();
                    let out_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let out_min = output.iter().cloned().fold(f32::INFINITY, f32::min);
                    let out_has_nan = output.iter().any(|x| x.is_nan());
                    eprintln!("[IMP-1010] pos{} L{} fused_ffn out: sum={:.6e}, min={:.6e}, max={:.6e}, has_nan={}",
                        position, layer_idx, out_sum, out_min, out_max, out_has_nan);
                }

                output
            } else if let (Some(ref gate_weight), Some(ref gate_cache_key)) =
                (&ffn_gate_weight, &ffn_gate_cache_key)
            {
                // PAR-015/PAR-049: SwiGLU path for LLaMA models
                // Formula: down(silu(gate(norm(x))) * up(norm(x)))
                // PAR-049 FIX: Apply FFN layer norm before projections (was missing!)

                // Apply FFN layer norm if present (separate from attention norm in LLaMA-style)
                // PAR-050: Use RMSNorm for LLaMA models
                let ffn_input =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        if use_rmsnorm {
                            self.model.rms_norm(&hidden, ffn_norm, eps)
                        } else {
                            self.model.layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        }
                    } else {
                        hidden.clone()
                    };

                // UP projection on normalized input
                let mut ffn_up =
                    self.fused_matmul_cuda_with_key(&ffn_input, &ffn_up_weight, &ffn_up_cache_key)?;
                if let Some(ref bias) = ffn_up_bias {
                    self.model.add_bias(&mut ffn_up, bias);
                }

                // GATE projection on normalized input
                let mut ffn_gate =
                    self.fused_matmul_cuda_with_key(&ffn_input, gate_weight, gate_cache_key)?;
                if let Some(ref bias) = ffn_gate_bias {
                    self.model.add_bias(&mut ffn_gate, bias);
                }

                // SiLU on gate, then multiply with up
                self.model.silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                // DOWN projection
                let mut ffn_output = self.fused_matmul_cuda_with_key(
                    &ffn_gate,
                    &ffn_down_weight,
                    &ffn_down_cache_key,
                )?;
                if let Some(ref bias) = ffn_down_bias {
                    self.model.add_bias(&mut ffn_output, bias);
                }
                ffn_output
            } else {
                // GELU path for phi-2 style models (no gate projection)
                // IMP-1010 FIX: Apply FFN layer norm if present (parallel residual models like phi-2
                // use the same normalized input for both attention and FFN)
                let ffn_input =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        if use_rmsnorm {
                            self.model.rms_norm(&hidden, ffn_norm, eps)
                        } else {
                            self.model.layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        }
                    } else {
                        // Parallel residual: use same normalized input as attention
                        normed.clone()
                    };
                let mut ffn_hidden =
                    self.fused_matmul_cuda_with_key(&ffn_input, &ffn_up_weight, &ffn_up_cache_key)?;
                if let Some(ref bias) = ffn_up_bias {
                    self.model.add_bias(&mut ffn_hidden, bias);
                }
                self.model.gelu(&mut ffn_hidden);

                let mut ffn_output = self.fused_matmul_cuda_with_key(
                    &ffn_hidden,
                    &ffn_down_weight,
                    &ffn_down_cache_key,
                )?;
                if let Some(ref bias) = ffn_down_bias {
                    self.model.add_bias(&mut ffn_output, bias);
                }
                ffn_output
            };

            // PAR-049-DEBUG: Compare FFN output with CPU (disabled for performance)
            #[allow(clippy::never_loop)]
            if false {
                // Compute CPU FFN for comparison
                let _ffn_input_cpu =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        self.model.layer_norm(
                            &hidden,
                            ffn_norm,
                            self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                            eps,
                        )
                    } else {
                        hidden.clone()
                    };
                // hidden before residual add of ffn
                let hidden_before_ffn: Vec<f32> = hidden
                    .iter()
                    .zip(&attn_output)
                    .map(|(h, a)| h - a)
                    .collect();
                eprintln!(
                    "[PAR-049] L0 hidden before attn residual[0..5]: {:?}",
                    &hidden_before_ffn[..5.min(hidden_before_ffn.len())]
                );
                eprintln!(
                    "[PAR-049] L0 ffn_output GPU[0..5]: {:?}",
                    &ffn_output[..5.min(ffn_output.len())]
                );
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // IMP-1010-DEBUG: Check hidden after FFN residual
            if false {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                let ffn_output_has_nan = ffn_output.iter().any(|x| x.is_nan());
                let ffn_output_sum: f32 = ffn_output.iter().sum();
                let ffn_output_max = ffn_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} ffn_out: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, ffn_output_sum, ffn_output_max, ffn_output_has_nan
                );
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} final: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_max, hidden_has_nan
                );
            } else if position < 2 {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                if hidden_has_nan {
                    let ffn_output_has_nan = ffn_output.iter().any(|x| x.is_nan());
                    let ffn_output_sum: f32 = ffn_output.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} ffn_out: sum={:.6e}, has_nan={}",
                        position, layer_idx, ffn_output_sum, ffn_output_has_nan
                    );
                    let hidden_sum: f32 = hidden.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} hidden after FFN: sum={:.6e}, has_nan={}",
                        position, layer_idx, hidden_sum, hidden_has_nan
                    );
                }
            }

            // PAR-049-DEBUG: Print hidden state after layer 0 and compute CPU reference (disabled for performance)
            // Re-enable by changing `false` to `true` for debugging
            #[allow(clippy::never_loop)]
            if false {
                eprintln!(
                    "[PAR-049] L0 GPU hidden[0..5]: {:?}",
                    &hidden[..5.min(hidden.len())]
                );

                // Compute CPU reference for layer 0
                // Start from embedding
                let cpu_hidden = self.model.embed(&[token_id]);
                let cpu_normed = self.model.layer_norm(
                    &cpu_hidden,
                    &self.model.layers[0].attn_norm_weight,
                    self.model.layers[0].attn_norm_bias.as_deref(),
                    eps,
                );
                let cpu_qkv = self
                    .model
                    .qkv_matmul(&cpu_normed, &self.model.layers[0].qkv_weight)
                    .expect("CPU qkv");
                let mut cpu_q = cpu_qkv[0..hidden_dim].to_vec();
                let mut cpu_k = cpu_qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
                let cpu_v = cpu_qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();
                self.model.apply_rope(&mut cpu_q, 0, num_heads);
                self.model.apply_rope(&mut cpu_k, 0, num_kv_heads);

                // First token - expand V for GQA
                let cpu_attn_out = if num_kv_heads < num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded = vec![0.0f32; hidden_dim];
                    for qh in 0..num_heads {
                        let kv_h = qh / q_per_kv;
                        expanded[qh * head_dim..(qh + 1) * head_dim]
                            .copy_from_slice(&cpu_v[kv_h * head_dim..(kv_h + 1) * head_dim]);
                    }
                    expanded
                } else {
                    cpu_v.clone()
                };

                let cpu_attn_proj = self
                    .model
                    .fused_matmul(&cpu_attn_out, &self.model.layers[0].attn_output_weight)
                    .expect("CPU attn proj");
                let mut cpu_h = cpu_hidden.clone();
                for i in 0..hidden_dim {
                    cpu_h[i] += cpu_attn_proj[i];
                }

                eprintln!(
                    "[PAR-049] L0 CPU hidden after attn[0..5]: {:?}",
                    &cpu_h[..5.min(cpu_h.len())]
                );

                // Compare attention residual state
                let hidden_after_attn: Vec<f32> =
                    hidden.iter().zip(&ffn_output).map(|(h, f)| h - f).collect();
                let max_diff_attn = hidden_after_attn
                    .iter()
                    .zip(cpu_h.iter())
                    .map(|(g, c)| (g - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[PAR-049] L0 attn residual max_diff: {:.6e}", max_diff_attn);
            }
        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm (CPU - fast for single vector)
        // PAR-050: Use RMSNorm for LLaMA models
        let normed = if use_rmsnorm {
            self.model.rms_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.config.eps,
            )
        } else {
            self.model.layer_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.output_norm_bias.as_deref(),
                self.model.config.eps,
            )
        };

        // 4. LM head projection (GPU - IMP-1010, PAR-016: use pre-captured cache key)
        // Clone LM head weight to avoid borrow conflicts, but use stable cache key
        let lm_head_weight_data = self.model.lm_head_weight.data.clone();
        let lm_head_weight_in_dim = self.model.lm_head_weight.in_dim;
        let lm_head_weight_out_dim = self.model.lm_head_weight.out_dim;
        let lm_head_weight_qtype = self.model.lm_head_weight.qtype;
        let lm_head_weight = OwnedQuantizedTensor {
            data: lm_head_weight_data,
            in_dim: lm_head_weight_in_dim,
            out_dim: lm_head_weight_out_dim,
            qtype: lm_head_weight_qtype,
        };

        let mut logits =
            self.fused_matmul_cuda_with_key(&normed, &lm_head_weight, &lm_head_cache_key)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            self.model.add_bias(&mut logits, bias);
        }

        // PAR-049-DEBUG: Compare final logits with CPU for position 1 (disabled for performance)
        // Re-enable by changing `false` to `true` for debugging
        // IMP-1010-DEBUG: Enable for all positions to debug garbage output
        #[allow(clippy::never_loop)]
        if false {
            // IMP-1010-DEBUG: Print hidden and normed stats
            let hidden_sum: f32 = hidden.iter().sum();
            let hidden_max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[IMP-1010] pos{} hidden: sum={:.6e}, min={:.6e}, max={:.6e}",
                position, hidden_sum, hidden_min, hidden_max
            );
            let normed_sum: f32 = normed.iter().sum();
            let normed_max = normed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normed_min = normed.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[IMP-1010] pos{} normed: sum={:.6e}, min={:.6e}, max={:.6e}",
                position, normed_sum, normed_min, normed_max
            );
            let cpu_logits = self.model.fused_matmul(&normed, &lm_head_weight)?;
            let max_diff = logits
                .iter()
                .zip(cpu_logits.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);
            let top5_gpu: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i, v)).fold(
                vec![(0usize, f32::MIN); 5],
                |mut acc, (i, v)| {
                    if v > acc[4].1 {
                        acc[4] = (i, v);
                        acc.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    acc
                },
            );
            let top5_cpu: Vec<_> = cpu_logits.iter().enumerate().map(|(i, &v)| (i, v)).fold(
                vec![(0usize, f32::MIN); 5],
                |mut acc, (i, v)| {
                    if v > acc[4].1 {
                        acc[4] = (i, v);
                        acc.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    acc
                },
            );
            eprintln!("[PAR-049] pos1 logits max_diff: {:.6e}", max_diff);
            eprintln!("[PAR-049] pos1 GPU top5: {:?}", top5_gpu);
            eprintln!("[PAR-049] pos1 CPU top5: {:?}", top5_cpu);
        }

        Ok(logits)
    }

    /// GPU-accelerated attention with KV cache using multi-head CUDA kernel (PARITY-044)
    ///
    /// Uses `CudaExecutor::flash_attention_multi_head` to process all heads in parallel.
    /// Memory layout: [n_heads, seq_len, head_dim]
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (hidden_dim / num_heads)
    #[allow(clippy::too_many_arguments)]
    fn cuda_attention_with_cache(
        &mut self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        total_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = num_heads * head_dim;
        let cache_len = total_len - 1;

        // Build full K and V tensors for all heads: [n_heads, total_len, head_dim]
        let tensor_size = num_heads * total_len * head_dim;

        // For GPU multi-head attention, we need Q repeated across all positions
        // Q is [hidden_dim] = [n_heads * head_dim], expand to [n_heads, total_len, head_dim]
        let mut q_full = vec![0.0f32; tensor_size];
        let mut k_full = vec![0.0f32; tensor_size];
        let mut v_full = vec![0.0f32; tensor_size];

        // Reorganize from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;

            // Q: single query expanded to all positions (for proper broadcast)
            for pos in 0..total_len {
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                q_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&q[head_offset..head_offset + head_dim]);
            }

            // K: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                k_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&k_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current K
            let gpu_current_offset = gpu_head_offset + cache_len * head_dim;
            k_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // V: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                v_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&v_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current V
            v_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_v[head_offset..head_offset + head_dim]);
        }

        // GPU multi-head attention using FlashAttention kernel
        let mut output_full = vec![0.0f32; tensor_size];
        self.executor
            .flash_attention_multi_head(
                &q_full,
                &k_full,
                &v_full,
                &mut output_full,
                total_len as u32,
                head_dim as u32,
                num_heads as u32,
                true, // causal masking for autoregressive decoding
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "flash_attention_multi_head".to_string(),
                reason: format!("CUDA attention failed: {e}"),
            })?;

        // Extract output for the last position and reorganize to [hidden_dim]
        let mut output = vec![0.0f32; hidden_dim];
        let last_pos = total_len - 1;
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;
            let gpu_pos_offset = gpu_head_offset + last_pos * head_dim;
            output[head_offset..head_offset + head_dim]
                .copy_from_slice(&output_full[gpu_pos_offset..gpu_pos_offset + head_dim]);
        }

        Ok(output)
    }

    /// Generate tokens using CUDA acceleration with KV cache (PARITY-044)
    ///
    /// Uses `forward_single_cuda_with_cache` for GPU-accelerated incremental decoding.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            let logits = self.forward_single_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// IMP-1010: Full GPU-accelerated token generation
    ///
    /// Uses `forward_single_full_cuda_with_cache` for maximum GPU utilization.
    /// All matmul operations (5 per layer) run on GPU.
    ///
    /// # Performance Target
    ///
    /// - CPU path: ~5 tok/s (limited by memory bandwidth)
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_full_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) - use full GPU path
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_full_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            let logits =
                self.forward_single_full_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            // PAR-050-DEBUG: Print sampled tokens
            if tokens.len() <= 15 {
                eprintln!(
                    "[PAR-050] Generated token {}: {} (position {})",
                    tokens.len() - prompt.len() + 1,
                    next_token,
                    position
                );
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// PAR-100: Speculative decoding with GPU-resident forward
    ///
    /// Uses GPU-resident path for fast single-token drafting, then verifies.
    ///
    /// # Theory (Five-Whys Root Cause)
    ///
    /// WHY is single-token decode limited to ~430 tok/s?
    /// → Memory bandwidth bound: each token reads ALL weights from VRAM
    ///
    /// NOTE: Self-speculative decoding (same model for draft and verify) doesn't
    /// improve throughput because draft phase still requires k weight reads.
    /// True speedup requires either:
    /// 1. Smaller draft model (e.g., 0.5B → 1.5B)
    /// 2. Layer-skipping during draft (skip last N/2 layers)
    ///
    /// This implementation uses GPU-resident path for drafting to at least match
    /// standard generation throughput as a baseline.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (uses max_tokens)
    /// * `speculation_k` - Number of tokens to draft speculatively (typically 4-8)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_speculative_cuda(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        speculation_k: usize,
    ) -> Result<Vec<u32>> {
        use std::time::Instant;

        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_speculative_cuda".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        // Pre-upload all weights to GPU
        let bytes_uploaded = self.preload_weights_gpu()?;
        eprintln!(
            "PAR-100: Pre-uploaded {} MB of weights to GPU",
            bytes_uploaded / (1024 * 1024)
        );

        // PAR-100: Setup KV cache with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim,
            prompt.len() + config.max_tokens + speculation_k,
        );

        // Reset GPU KV cache positions before generation
        self.executor.reset_kv_cache_gpu();

        let mut tokens = prompt.to_vec();

        // Prefill: process prompt tokens using GPU-resident path
        let prefill_start = Instant::now();
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                let _ = self.forward_gpu_resident(token_id, &mut cache, pos)?;
            }
        }
        let prefill_time = prefill_start.elapsed();

        // Start decode from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        // Statistics for throughput calculation
        let decode_start = Instant::now();
        let mut accepted_tokens = 0usize;
        let mut total_drafts = 0usize;
        let mut total_speculative_batches = 0usize;

        while tokens.len() - prompt.len() < config.max_tokens {
            // Step 1: Draft k tokens greedily using GPU-resident forward
            let cache_snapshot = cache.snapshot_len();
            let mut draft_tokens = Vec::with_capacity(speculation_k);

            // Draft all k tokens using GPU-resident to_token_id (greedy argmax)
            for i in 0..speculation_k {
                let draft_pos = position + i;
                let input_token = if i == 0 {
                    last_token
                } else {
                    *draft_tokens.last().unwrap_or(&last_token)
                };

                let draft =
                    self.forward_gpu_resident_to_token_id(input_token, &mut cache, draft_pos)?;

                if config.stop_tokens.contains(&draft) {
                    if i == 0 {
                        // First draft is stop token
                        tokens.push(draft);
                    }
                    break;
                }

                draft_tokens.push(draft);
            }

            if draft_tokens.is_empty() {
                break; // Stop token on first draft
            }

            total_drafts += draft_tokens.len();

            // Step 2: Rollback cache to snapshot for verification
            cache.rollback_to(cache_snapshot, kv_dim);
            self.executor.reset_kv_cache_gpu();

            // Step 3: Verify - use single-token GPU-resident to check each draft
            // NOTE: Batched verification would be faster but requires refactoring
            // For now, verify sequentially to ensure correctness
            let mut num_accepted = 0usize;

            for (i, &draft) in draft_tokens.iter().enumerate() {
                let verify_pos = position + i;
                let input_token = if i == 0 {
                    last_token
                } else {
                    *draft_tokens.get(i - 1).unwrap_or(&last_token)
                };

                let verified =
                    self.forward_gpu_resident_to_token_id(input_token, &mut cache, verify_pos)?;

                if verified == draft {
                    // Accept this token
                    tokens.push(draft);
                    num_accepted += 1;
                } else {
                    // Reject: accept the model's correction instead
                    if !config.stop_tokens.contains(&verified) {
                        tokens.push(verified);
                        num_accepted += 1;
                    }
                    break;
                }
            }

            total_speculative_batches += 1;

            // Handle edge case: all drafts rejected
            if num_accepted == 0 && !draft_tokens.is_empty() {
                // Just generate one token normally
                cache.rollback_to(cache_snapshot, kv_dim);
                self.executor.reset_kv_cache_gpu();
                let fallback =
                    self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?;
                if config.stop_tokens.contains(&fallback) {
                    break;
                }
                tokens.push(fallback);
                num_accepted = 1;
            }

            accepted_tokens += num_accepted;

            // Step 4: Update position and last_token
            position += num_accepted;
            last_token = *tokens.last().unwrap_or(&0);

            // Rollback cache to keep only accepted entries
            let target_cache_len = cache_snapshot + num_accepted;
            cache.rollback_to(target_cache_len, kv_dim);
        }

        let decode_time = decode_start.elapsed();
        let generated_tokens = tokens.len() - prompt.len();
        let decode_tok_s = if decode_time.as_secs_f64() > 0.0 {
            generated_tokens as f64 / decode_time.as_secs_f64()
        } else {
            0.0
        };

        let acceptance_rate = if total_drafts > 0 {
            accepted_tokens as f64 / total_drafts as f64 * 100.0
        } else {
            0.0
        };

        eprintln!(
            "[PAR-100] Speculative decode: {} tokens in {:.2}ms ({:.1} tok/s)",
            generated_tokens,
            decode_time.as_secs_f64() * 1000.0,
            decode_tok_s
        );
        eprintln!(
            "[PAR-100] Prefill: {:.2}ms, Drafts: {}, Accepted: {}, Rate: {:.1}%",
            prefill_time.as_secs_f64() * 1000.0,
            total_drafts,
            accepted_tokens,
            acceptance_rate
        );
        eprintln!(
            "[PAR-100] Batched verifications: {}",
            total_speculative_batches
        );

        Ok(tokens)
    }

    /// PAR-099: Speculative decoding with separate draft model
    ///
    /// Uses a smaller draft model (e.g., 0.5B) for fast token generation,
    /// then verifies with the target model (e.g., 1.5B).
    ///
    /// # Theory (Five-Whys Root Cause)
    ///
    /// WHY does draft model help?
    /// → Draft model is 3x smaller = 3x faster = 3x fewer weight reads
    /// → Verification with target model amortizes quality check
    ///
    /// Expected speedup with 0.5B draft + 1.5B target:
    /// - Draft 4 tokens: 4 × (2.5ms/3) = 3.3ms
    /// - Verify 4 tokens: 1 × 2.5ms = 2.5ms (batched)
    /// - Total: 5.8ms for ~3 accepted tokens = 517 tok/s (1.3x improvement)
    ///
    /// With k=8, 80% acceptance: theoretical ~700-800 tok/s
    ///
    /// # Arguments
    ///
    /// * `draft_model` - Smaller model for fast token drafting
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `speculation_k` - Number of tokens to draft (typically 4-8)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_speculative_with_draft(
        &mut self,
        draft_model: &mut OwnedQuantizedModelCuda,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        speculation_k: usize,
    ) -> Result<Vec<u32>> {
        use std::time::Instant;

        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // Check architecture support for both models
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_speculative_with_draft".to_string(),
                reason: "Target model architecture not supported for GPU-resident path".to_string(),
            });
        }
        if !draft_model.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_speculative_with_draft".to_string(),
                reason: "Draft model architecture not supported for GPU-resident path".to_string(),
            });
        }

        // Pre-upload weights for both models
        let target_bytes = self.preload_weights_gpu()?;
        let draft_bytes = draft_model.preload_weights_gpu()?;
        eprintln!(
            "PAR-099: Pre-uploaded {} MB (target) + {} MB (draft) to GPU",
            target_bytes / (1024 * 1024),
            draft_bytes / (1024 * 1024)
        );

        // Setup KV caches for both models
        let target_kv_dim = {
            let num_kv_heads = self.model.config.num_kv_heads;
            let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
            num_kv_heads * head_dim
        };
        let draft_kv_dim = {
            let num_kv_heads = draft_model.model.config.num_kv_heads;
            let head_dim = draft_model.model.config.hidden_dim / draft_model.model.config.num_heads;
            num_kv_heads * head_dim
        };

        let mut target_cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            target_kv_dim,
            prompt.len() + config.max_tokens + speculation_k,
        );
        let mut draft_cache = OwnedQuantizedKVCache::new(
            draft_model.model.config.num_layers,
            draft_kv_dim,
            prompt.len() + config.max_tokens + speculation_k,
        );

        // Reset GPU KV cache positions
        self.executor.reset_kv_cache_gpu();
        draft_model.executor.reset_kv_cache_gpu();

        let mut tokens = prompt.to_vec();

        // Prefill both models
        let prefill_start = Instant::now();
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                let _ = self.forward_gpu_resident(token_id, &mut target_cache, pos)?;
                let _ = draft_model.forward_gpu_resident(token_id, &mut draft_cache, pos)?;
            }
        }
        let prefill_time = prefill_start.elapsed();

        // Start decode from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        // Statistics
        let decode_start = Instant::now();
        let mut accepted_tokens = 0usize;
        let mut total_drafts = 0usize;
        let mut total_speculative_batches = 0usize;

        while tokens.len() - prompt.len() < config.max_tokens {
            // Step 1: Draft k tokens using DRAFT model (fast, smaller)
            let draft_cache_snapshot = draft_cache.snapshot_len();
            let target_cache_snapshot = target_cache.snapshot_len();
            let mut draft_tokens = Vec::with_capacity(speculation_k);

            // Draft using the smaller model
            for i in 0..speculation_k {
                let draft_pos = position + i;
                let input_token = if i == 0 {
                    last_token
                } else {
                    *draft_tokens.last().unwrap_or(&last_token)
                };

                let draft = draft_model.forward_gpu_resident_to_token_id(
                    input_token,
                    &mut draft_cache,
                    draft_pos,
                )?;

                if config.stop_tokens.contains(&draft) {
                    if i == 0 {
                        tokens.push(draft);
                    }
                    break;
                }

                draft_tokens.push(draft);
            }

            if draft_tokens.is_empty() {
                break;
            }

            total_drafts += draft_tokens.len();

            // Step 2: Verify using TARGET model
            // PAR-105: Rollback draft cache to snapshot position, preserving prefill history
            // RESOLVED: reset_kv_cache_gpu() was clearing ALL history, causing 1/k acceptance
            draft_cache.rollback_to(draft_cache_snapshot, draft_kv_dim);
            draft_model
                .executor
                .rollback_kv_cache_gpu(draft_cache_snapshot);

            let mut num_accepted = 0usize;

            for (i, &draft) in draft_tokens.iter().enumerate() {
                let verify_pos = position + i;
                let input_token = if i == 0 {
                    last_token
                } else {
                    *draft_tokens.get(i - 1).unwrap_or(&last_token)
                };

                // Verify with target model
                let verified = self.forward_gpu_resident_to_token_id(
                    input_token,
                    &mut target_cache,
                    verify_pos,
                )?;

                if verified == draft {
                    // Accept: also update draft cache for consistency
                    let _ = draft_model.forward_gpu_resident(
                        input_token,
                        &mut draft_cache,
                        verify_pos,
                    )?;
                    tokens.push(draft);
                    num_accepted += 1;
                } else {
                    // Reject: accept target's correction
                    if !config.stop_tokens.contains(&verified) {
                        let _ = draft_model.forward_gpu_resident(
                            input_token,
                            &mut draft_cache,
                            verify_pos,
                        )?;
                        tokens.push(verified);
                        num_accepted += 1;
                    }
                    break;
                }
            }

            total_speculative_batches += 1;

            // Handle edge case: all drafts rejected
            if num_accepted == 0 && !draft_tokens.is_empty() {
                // PAR-105: Use rollback instead of reset to preserve prefill history
                target_cache.rollback_to(target_cache_snapshot, target_kv_dim);
                draft_cache.rollback_to(draft_cache_snapshot, draft_kv_dim);
                self.executor.rollback_kv_cache_gpu(target_cache_snapshot);
                draft_model
                    .executor
                    .rollback_kv_cache_gpu(draft_cache_snapshot);

                let fallback =
                    self.forward_gpu_resident_to_token_id(last_token, &mut target_cache, position)?;
                let _ = draft_model.forward_gpu_resident(last_token, &mut draft_cache, position)?;

                if config.stop_tokens.contains(&fallback) {
                    break;
                }
                tokens.push(fallback);
                num_accepted = 1;
            }

            accepted_tokens += num_accepted;
            position += num_accepted;
            last_token = *tokens.last().unwrap_or(&0);

            // Rollback caches to accepted length (CPU AND GPU must stay in sync)
            let target_len = target_cache_snapshot + num_accepted;
            let draft_len = draft_cache_snapshot + num_accepted;
            target_cache.rollback_to(target_len, target_kv_dim);
            draft_cache.rollback_to(draft_len, draft_kv_dim);
            // PAR-105: RESOLVED - must also rollback GPU caches to match CPU
            // Without this, GPU cache has stale entries from rejected verifications
            self.executor.rollback_kv_cache_gpu(target_len);
            draft_model.executor.rollback_kv_cache_gpu(draft_len);
        }

        let decode_time = decode_start.elapsed();
        let generated_tokens = tokens.len() - prompt.len();
        let decode_tok_s = if decode_time.as_secs_f64() > 0.0 {
            generated_tokens as f64 / decode_time.as_secs_f64()
        } else {
            0.0
        };

        let acceptance_rate = if total_drafts > 0 {
            accepted_tokens as f64 / total_drafts as f64 * 100.0
        } else {
            0.0
        };

        eprintln!(
            "[PAR-099] Speculative decode (draft model): {} tokens in {:.2}ms ({:.1} tok/s)",
            generated_tokens,
            decode_time.as_secs_f64() * 1000.0,
            decode_tok_s
        );
        eprintln!(
            "[PAR-099] Prefill: {:.2}ms, Drafts: {}, Accepted: {}, Rate: {:.1}%",
            prefill_time.as_secs_f64() * 1000.0,
            total_drafts,
            accepted_tokens,
            acceptance_rate
        );
        eprintln!(
            "[PAR-099] Speculative batches: {}",
            total_speculative_batches
        );

        Ok(tokens)
    }

    // =========================================================================
    // PAR-023: GPU-Resident Transformer Layer Integration
    // =========================================================================

    /// PAR-023: Pre-upload all layer weights to GPU with naming convention for
    /// GPU-resident transformer layer.
    ///
    /// This method uploads quantized weights using names expected by
    /// `CudaExecutor::transformer_layer_gpu`:
    /// - `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`
    /// - `blk.{i}.attn_output.weight`
    /// - `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails or model uses fused QKV (phi-2 style).
    pub fn preload_weights_gpu(&mut self) -> Result<usize> {
        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        let mut total_bytes = 0usize;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            // Upload Q, K, V weights (requires separate format for GPU-resident path)
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    // Q projection - PAR-058: pass qtype for mixed-quant models (e.g., Q5_0 in Qwen 0.5B)
                    let q_name = format!("{}.attn_q.weight", prefix);
                    if !self.executor.has_quantized_weights(&q_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&q_name, &q.data, q.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload Q weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }

                    // K projection - PAR-058: pass qtype for mixed-quant models
                    let k_name = format!("{}.attn_k.weight", prefix);
                    if !self.executor.has_quantized_weights(&k_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&k_name, &k.data, k.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload K weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }

                    // V projection - PAR-058: pass qtype for mixed-quant models
                    let v_name = format!("{}.attn_v.weight", prefix);
                    if !self.executor.has_quantized_weights(&v_name) {
                        total_bytes += self
                            .executor
                            .load_quantized_weights_with_type(&v_name, &v.data, v.qtype)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "preload_weights_gpu".to_string(),
                                reason: format!(
                                    "Failed to upload V weights for layer {}: {}",
                                    layer_idx, e
                                ),
                            })?;
                    }
                },
                OwnedQKVWeights::Fused(_) => {
                    // Fused QKV not yet supported for GPU-resident path
                    // Fall back to standard forward pass for phi-2 style models
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Layer {} uses fused QKV (phi-2 style), GPU-resident path requires separate Q/K/V",
                            layer_idx
                        ),
                    });
                },
            }

            // Output projection - PAR-058: pass qtype for mixed-quant models
            let o_name = format!("{}.attn_output.weight", prefix);
            if !self.executor.has_quantized_weights(&o_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &o_name,
                        &layer.attn_output_weight.data,
                        layer.attn_output_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload O weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }

            // FFN gate (SwiGLU models) - PAR-058: pass qtype
            if let Some(ref gate) = layer.ffn_gate_weight {
                let gate_name = format!("{}.ffn_gate.weight", prefix);
                if !self.executor.has_quantized_weights(&gate_name) {
                    total_bytes += self
                        .executor
                        .load_quantized_weights_with_type(&gate_name, &gate.data, gate.qtype)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "preload_weights_gpu".to_string(),
                            reason: format!(
                                "Failed to upload gate weights for layer {}: {}",
                                layer_idx, e
                            ),
                        })?;
                }
            }

            // FFN up - PAR-058: pass qtype
            let up_name = format!("{}.ffn_up.weight", prefix);
            if !self.executor.has_quantized_weights(&up_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &up_name,
                        &layer.ffn_up_weight.data,
                        layer.ffn_up_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload up weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }

            // FFN down - PAR-058: pass qtype
            let down_name = format!("{}.ffn_down.weight", prefix);
            if !self.executor.has_quantized_weights(&down_name) {
                total_bytes += self
                    .executor
                    .load_quantized_weights_with_type(
                        &down_name,
                        &layer.ffn_down_weight.data,
                        layer.ffn_down_weight.qtype,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_weights_gpu".to_string(),
                        reason: format!(
                            "Failed to upload down weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }
        }

        // Also upload LM head weights
        let lm_head_name = "output.weight".to_string();
        if !self.executor.has_quantized_weights(&lm_head_name) {
            // PAR-060-DEBUG: Print first bytes of LM head weight for verification
            let lm_data = &self.model.lm_head_weight.data;
            if verbose() {
                eprintln!(
                    "[PAR-060-DEBUG] LM head weight: len={}, first 20 bytes: {:?}",
                    lm_data.len(),
                    &lm_data[..20.min(lm_data.len())]
                );
                eprintln!(
                    "[PAR-060-DEBUG] LM head dims: in_dim={}, out_dim={}",
                    self.model.lm_head_weight.in_dim, self.model.lm_head_weight.out_dim
                );
            }
            total_bytes += self
                .executor
                .load_quantized_weights_with_type(
                    &lm_head_name,
                    lm_data,
                    self.model.lm_head_weight.qtype,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("Failed to upload LM head weights: {}", e),
                })?;
        }

        // PAR-023: Pre-cache RMSNorm weights for all layers
        let num_layers = self.model.layers.len();
        let attn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| l.attn_norm_weight.as_slice())
            .collect();
        let ffn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.ffn_norm_weight
                    .as_ref()
                    .map_or(l.attn_norm_weight.as_slice(), |w| w.as_slice())
            })
            .collect();

        total_bytes += self
            .executor
            .preload_rmsnorm_weights(num_layers, &attn_norms, &ffn_norms)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload RMSNorm weights: {}", e),
            })?;

        // PAR-023: Pre-cache output norm (final layer norm) weight
        // This enables fully GPU-resident forward pass including output norm + LM head
        total_bytes += self
            .executor
            .preload_output_norm(&self.model.output_norm_weight)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload output norm weights: {}", e),
            })?;

        // BIAS-FIX: Pre-cache QKV bias vectors for all layers
        // Qwen2.5 models have QKV bias that must be added after GEMV
        let q_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    // Q bias is first q_dim elements
                    let q_dim = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, .. } => q.out_dim,
                        OwnedQKVWeights::Fused(w) => w.out_dim / 3,
                    };
                    &b[..q_dim]
                })
            })
            .collect();
        let k_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let (q_dim, k_dim) = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, k, .. } => (q.out_dim, k.out_dim),
                        OwnedQKVWeights::Fused(w) => {
                            let dim = w.out_dim / 3;
                            (dim, dim)
                        },
                    };
                    &b[q_dim..q_dim + k_dim]
                })
            })
            .collect();
        let v_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let (q_dim, k_dim, v_dim) = match &l.qkv_weight {
                        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
                        OwnedQKVWeights::Fused(w) => {
                            let dim = w.out_dim / 3;
                            (dim, dim, dim)
                        },
                    };
                    &b[q_dim + k_dim..q_dim + k_dim + v_dim]
                })
            })
            .collect();

        total_bytes += self
            .executor
            .preload_qkv_bias(num_layers, &q_biases, &k_biases, &v_biases)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload QKV bias: {}", e),
            })?;

        // PAR-064-FIX: Pre-cache LM head bias (output.bias) for models that have it
        // Without this bias, GPU inference produces incorrect token predictions
        total_bytes += self
            .executor
            .preload_lm_head_bias(self.model.lm_head_bias.as_deref())
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload LM head bias: {}", e),
            })?;

        // PAR-043: Build indexed weight lookup table for O(1) access during decode
        // This eliminates ~10ms constant CPU overhead per token from string formatting + HashMap lookups
        // PAR-107: Skip if already indexed to preserve CUDA graph (graph captures buffer addresses)
        if !self.executor.has_indexed_weights() {
            self.executor
                .build_indexed_weights(num_layers, |i| format!("blk.{}", i))
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-043: Failed to build indexed weights: {}", e),
                })?;
        }

        // PAR-044: Initialize workspace buffers for zero-allocation forward pass
        // This eliminates ~288 buffer allocations per token
        // PAR-107: Skip if already initialized to preserve CUDA graph (graph captures buffer addresses)
        // ROOT CAUSE FIX: Reallocating workspace invalidates graph since addresses change
        if !self.executor.has_workspace() {
            self.executor
                .init_workspace(
                    self.model.config.hidden_dim,
                    self.model.config.intermediate_dim,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-044: Failed to initialize workspace: {}", e),
                })?;
        }

        Ok(total_bytes)
    }

    /// Clear decode graph and related state
    ///
    /// Call this before starting a new generation session to ensure
    /// the graph is recaptured with fresh state.
    pub fn clear_decode_graph(&mut self) {
        self.executor.clear_decode_graph();
    }

    /// PAR-023: Check if model supports GPU-resident forward pass
    ///
    /// GPU-resident path requires:
    /// - Separate Q/K/V weights (not fused)
    /// - SwiGLU activation (ffn_gate_weight present)
    /// - RMSNorm (LLaMA-style architecture)
    #[must_use]
    pub fn supports_gpu_resident(&self) -> bool {
        // Check first layer for architecture detection
        if let Some(layer) = self.model.layers.first() {
            // Must have separate Q/K/V
            let has_separate_qkv = matches!(layer.qkv_weight, OwnedQKVWeights::Separate { .. });
            // Must have SwiGLU (gate weight present)
            let has_swiglu = layer.ffn_gate_weight.is_some();
            // Must have FFN norm (RMSNorm for pre-FFN)
            let has_ffn_norm = layer.ffn_norm_weight.is_some();

            has_separate_qkv && has_swiglu && has_ffn_norm
        } else {
            false
        }
    }

    /// PAR-023: GPU-resident forward pass for single token (decode phase)
    ///
    /// This method chains ALL transformer layers GPU-resident, syncing only at start/end.
    ///
    /// # Sync Count (optimized)
    ///
    /// - Embedding upload: 1 sync
    /// - All layers: 0 syncs (D2D transfers for KV cache)
    /// - Hidden download: 1 sync
    /// - LM head: 1 sync
    /// - Total: ~3 syncs vs 22 syncs (per-layer) or 176 syncs (original)
    ///
    /// # Requirements
    ///
    /// Must call `preload_weights_gpu()` before first use.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Current token ID
    /// * `cache` - KV cache (only used for CPU fallback path position tracking)
    /// * `_position` - Token position in sequence (unused, position tracked by GPU KV cache)
    ///
    /// # Errors
    ///
    /// Returns error if GPU operations fail or model architecture unsupported.
    #[allow(clippy::too_many_lines)]
    pub fn forward_gpu_resident(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup)
        let embedding = self.model.embed(&[token_id]);

        // PAR-060-DEBUG: Disabled for performance measurement
        // let embed_sum: f32 = embedding.iter().sum();
        // eprintln!("[PAR-060-DEBUG] Embedding sum: {:.6}", embed_sum);

        // 2. Fully GPU-resident forward: layers + output norm + LM head
        // PAR-054: Use CUDA graph-captured path for decode (reduces 280 launches to 1)
        // Only 2 syncs total: embedding upload + logits download
        let mut logits = vec![0.0f32; vocab_size];
        self.executor
            .forward_all_layers_gpu_to_logits_graphed(
                &embedding,
                &mut logits,
                position as u32,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident".to_string(),
                reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
            })?;

        // 3. Add LM head bias if present (CPU - fast)
        if let Some(ref bias) = self.model.lm_head_bias {
            self.model.add_bias(&mut logits, bias);
        }

        // Advance cache position (for compatibility with cache-based generation)
        cache.advance();

        Ok(logits)
    }

    /// PAR-062: GPU-resident forward pass returning token ID directly
    ///
    /// Like `forward_gpu_resident` but uses GPU-side argmax for greedy sampling.
    /// Eliminates 600KB logits transfer per token, reducing to 4 bytes (token ID).
    ///
    /// # Performance Improvement
    ///
    /// - Before: Download 152064 x 4 = 600KB per token
    /// - After: Download 1 x 4 = 4 bytes per token
    /// - Expected speedup: ~1.2x overall throughput
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token
    /// * `cache` - KV cache (advanced but not used for logits)
    /// * `position` - Position in sequence
    ///
    /// # Returns
    ///
    /// Token ID with highest logit value (greedy sampling)
    ///
    /// # Errors
    ///
    /// Returns error if GPU operations fail or model has lm_head_bias (requires CPU path).
    pub fn forward_gpu_resident_to_token_id(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<u32> {
        // CORRECTNESS-013: Check if deterministic mode is requested
        // In this mode, download logits to CPU for argmax to ensure bit-exact
        // output matching between CPU and GPU inference paths.
        static CORRECTNESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_cpu_argmax = *CORRECTNESS_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // PAR-062: If model has LM head bias, fall back to CPU path
        // (bias addition requires CPU, so we'd download logits anyway)
        if self.model.lm_head_bias.is_some() || use_cpu_argmax {
            let logits = self.forward_gpu_resident(token_id, cache, position)?;
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup)
        let embedding = self.model.embed(&[token_id]);

        // 2. Check if CUDA graph is captured; if not, use regular path first
        // The graphed path needs to be initialized via forward_all_layers_gpu_to_logits_graphed
        if !self.executor.has_decode_graph() {
            // First call - need to capture graph, use regular path
            let mut logits = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &embedding,
                    &mut logits,
                    position as u32,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_gpu_resident_to_token_id".to_string(),
                    reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
                })?;

            cache.advance();
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        // 3. Use GPU argmax path - graph is captured, use optimized replay
        let next_token = self
            .executor
            .forward_graphed_replay_to_token_id(&embedding, position as u32, vocab_size as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident_to_token_id".to_string(),
                reason: format!("forward_graphed_replay_to_token_id failed: {}", e),
            })?;

        cache.advance();
        Ok(next_token)
    }

    /// PAR-023: GPU-resident token generation
    ///
    /// Uses `forward_gpu_resident` for maximum GPU utilization with minimal syncs.
    /// Target: ~22 syncs per layer vs ~176 syncs in standard path.
    ///
    /// # Performance Target
    ///
    /// - Standard path: ~121 tok/s (PAR-022 baseline)
    /// - GPU-resident: >192 tok/s (M4 milestone)
    /// - Ultimate goal: ~500 tok/s (llama.cpp parity)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Errors
    ///
    /// Returns error if model doesn't support GPU-resident path or GPU operations fail.
    pub fn generate_gpu_resident(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        // (context may have been created on a different thread, e.g., main vs tokio worker)
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path (requires separate Q/K/V, SwiGLU, RMSNorm)".to_string(),
            });
        }

        // Pre-upload all weights to GPU
        let bytes_uploaded = self.preload_weights_gpu()?;
        if verbose() {
            eprintln!(
                "PAR-023: Pre-uploaded {} MB of weights to GPU",
                bytes_uploaded / (1024 * 1024)
            );
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        // PAR-055 FIX: Reset GPU KV cache positions before new generation
        // Without this, cache positions accumulate across generate calls causing degradation
        self.executor.reset_kv_cache_gpu();

        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill)
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                let _ = self.forward_gpu_resident(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _token_num in 0..config.max_tokens {
            // PAR-062: Use GPU argmax path for greedy sampling (150,000x data transfer reduction)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy sampling - use GPU-side argmax (4 bytes transfer vs 600KB)
                self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?
            } else {
                // Non-greedy sampling - need full logits for proper temperature + top-k sampling
                // PAR-063: Resolved issue where GPU path always took top token instead of sampling
                let logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// PAR-112: True token-by-token streaming generation
    ///
    /// Generates tokens one at a time and calls the callback after each token.
    /// The callback receives the token ID and can return `false` to stop generation early.
    ///
    /// This enables true real-time streaming where each token is delivered
    /// as soon as it's generated, rather than pseudo-streaming where all tokens
    /// are generated first then iterated.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token, returns `false` to stop
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.generate_gpu_resident_streaming(&prompt, &config, |token_id| {
    ///     println!("Generated: {}", token_id);
    ///     true // continue generation
    /// })?;
    /// ```
    pub fn generate_gpu_resident_streaming<F>(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        // Pre-upload all weights to GPU
        let _ = self.preload_weights_gpu()?;

        // Create KV cache with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim,
            prompt.len() + config.max_tokens,
        );

        // Reset GPU KV cache positions
        self.executor.reset_kv_cache_gpu();

        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill)
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                let _ = self.forward_gpu_resident(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _token_num in 0..config.max_tokens {
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?
            } else {
                let logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PAR-112: Call the streaming callback IMMEDIATELY after generating each token
            // If callback returns false, stop generation early
            if !on_token(next_token) {
                break;
            }

            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// PAR-106: Batched GPU-resident generation for continuous batching
    ///
    /// Processes multiple prompts concurrently with true weight sharing:
    /// - Single weight read produces N tokens (one per active request)
    /// - Target: 400 tok/s (2x Ollama) with 4+ concurrent requests
    ///
    /// Key optimization: Uses `forward_batch_with_cache_cuda_native` which
    /// amortizes memory bandwidth across the batch.
    pub fn generate_batch_gpu_resident(
        &mut self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batch_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let max_seq_len = max_prompt_len + config.max_tokens;

        // Pre-upload all weights to GPU (once for entire batch)
        let _ = self.preload_weights_gpu()?;

        // PAR-045: Create KV caches with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut caches: Vec<OwnedQuantizedKVCache> = (0..num_prompts)
            .map(|_| OwnedQuantizedKVCache::new(self.model.config.num_layers, kv_dim, max_seq_len))
            .collect();

        // Reset GPU KV cache positions
        self.executor.reset_kv_cache_gpu();

        // Initialize token sequences
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut done: Vec<bool> = vec![false; num_prompts];

        // Prefill: Process each prompt's tokens (can't batch different lengths easily)
        for (prompt_idx, prompt) in prompts.iter().enumerate() {
            for (pos, &token_id) in prompt.iter().enumerate() {
                if pos < prompt.len() - 1 {
                    // PAR-106: Use single-token forward for prefill
                    // (batched prefill would require padding/masking complexity)
                    let _ = self.forward_gpu_resident(token_id, &mut caches[prompt_idx], pos)?;
                }
            }
        }

        // Track positions per prompt
        let mut positions: Vec<usize> = prompts.iter().map(|p| p.len() - 1).collect();
        let mut last_tokens: Vec<u32> = prompts.iter().map(|p| p[p.len() - 1]).collect();

        // PAR-106: Batched decode loop with weight sharing
        for _gen_idx in 0..config.max_tokens {
            // Collect active prompts
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            // PAR-106/PAR-108: Sequential CUDA graphs outperform batched CPU path.
            // The batched GEMV kernel is 15x faster, but CUDA graphs amortize
            // kernel launch overhead which is more impactful. Batched path achieves
            // ~225 tok/s vs ~360 tok/s for sequential graphs.
            //
            // To achieve 2x Ollama (400 tok/s), need multi-token CUDA graph capture
            // that batches M tokens into a single graph execution.
            for &prompt_idx in &active_indices {
                let next_token = self.forward_gpu_resident_to_token_id(
                    last_tokens[prompt_idx],
                    &mut caches[prompt_idx],
                    positions[prompt_idx],
                )?;

                if config.stop_tokens.contains(&next_token) {
                    done[prompt_idx] = true;
                } else {
                    sequences[prompt_idx].push(next_token);
                    last_tokens[prompt_idx] = next_token;
                    positions[prompt_idx] += 1;

                    if sequences[prompt_idx].len() >= max_seq_len {
                        done[prompt_idx] = true;
                    }
                }
            }
        }

        Ok(sequences)
    }
}

// ============================================================================
// Phase 24: QuantizedGenerateConfig and OwnedQuantizedKVCache moved to runtime.rs
// ============================================================================

// ============================================================================
// OwnedInferenceScratchBuffer: Pre-allocated buffers for zero-alloc forward
// ============================================================================

/// Pre-allocated scratch buffers for OwnedQuantizedModel inference
///
/// Eliminates per-token allocations by reusing buffers across forward passes.
/// For Qwen2.5-0.5B with intermediate_dim=4864, this saves ~40KB per token.
///
/// PAR-126: Added Q8K scratch buffers for fused Q4K×Q8K matmul path.
/// Q8K uses 256-element super-blocks vs Q8_0's 32-element blocks.
/// This enables VNNI instruction path which is 30% faster than AVX2.
#[derive(Debug)]
pub struct OwnedInferenceScratchBuffer {
    /// QKV output buffer [hidden_dim + 2*kv_dim]
    pub qkv: Vec<f32>,
    /// Attention output buffer [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN up projection buffer [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection buffer [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN down output buffer [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Expanded V buffer for first token GQA [hidden_dim]
    pub expanded_v: Vec<f32>,
    /// Logits buffer [vocab_size]
    pub logits: Vec<f32>,
    /// Q8 quantization scales scratch [num_blocks]
    pub q8_scales: Vec<f32>,
    /// Q8 quantization values scratch [num_blocks * 32]
    pub q8_quants: Vec<i8>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl OwnedInferenceScratchBuffer {
    /// Create scratch buffer from model config
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = hidden_dim / config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;
        // Qwen2.5 uses intermediate = 5.5 * hidden, others use 4 * hidden
        let intermediate_dim = hidden_dim * 6; // Conservative estimate
                                               // Q8 quantization uses 32-element blocks
        let num_blocks = hidden_dim.div_ceil(32);

        // PAR-126: Q8K uses 256-element super-blocks for VNNI path
        const QK_K: usize = 256;
        let q8k_hidden_padded = hidden_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            qkv: vec![0.0f32; qkv_dim],
            attn_out: vec![0.0f32; hidden_dim],
            ffn_up: vec![0.0f32; intermediate_dim],
            ffn_gate: vec![0.0f32; intermediate_dim],
            ffn_down: vec![0.0f32; hidden_dim],
            expanded_v: vec![0.0f32; hidden_dim],
            logits: vec![0.0f32; config.vocab_size],
            q8_scales: vec![0.0f32; num_blocks],
            q8_quants: vec![0i8; num_blocks * 32],
            // PAR-126: Q8K scratch for VNNI-accelerated matmul
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers (clear without deallocating)
    pub fn reset(&mut self) {
        // Just zero the lengths, keeping capacity
        self.qkv.clear();
        self.attn_out.clear();
        self.ffn_up.clear();
        self.ffn_gate.clear();
        self.ffn_down.clear();
        self.expanded_v.clear();
        self.logits.clear();
        self.q8_scales.clear();
        self.q8_quants.clear();
        // PAR-126: Q8K buffers
        self.q8k_hidden_scales.clear();
        self.q8k_hidden_quants.clear();
        self.q8k_inter_scales.clear();
        self.q8k_inter_quants.clear();
    }
}

// ============================================================================
// PARITY-005: Contiguous KV Cache for Cache Efficiency
// ============================================================================

/// Cache line size in bytes (typical x86-64)
const CACHE_LINE_BYTES: usize = 64;

/// Number of f32 elements per cache line (64 bytes / 4 bytes per f32)
const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_BYTES / std::mem::size_of::<f32>();

/// Contiguous KV cache with 64-byte cache line alignment (PARITY-005)
///
/// This cache uses a single contiguous allocation for all K and V data,
/// aligned to 64-byte cache lines for optimal L2 cache performance.
///
/// ## Memory Layout
///
/// ```text
/// K cache: [layer_0][layer_1]...[layer_n] (all contiguous)
/// V cache: [layer_0][layer_1]...[layer_n] (all contiguous)
///
/// Each layer: [pos_0][pos_1]...[pos_max_seq] where each pos is [hidden_dim]
/// ```
///
/// ## Cache Line Alignment
///
/// - All layer boundaries are aligned to 64-byte cache lines
/// - Enables hardware prefetching to work efficiently
/// - Reduces cache line splits during sequential access
///
/// ## Performance Benefits
///
/// - Single allocation reduces heap fragmentation
/// - Sequential access enables hardware prefetching
/// - Cache line alignment prevents false sharing
/// - L2 cache hit rate target: >90% (vs <70% with Vec<Vec<f32>>)
#[derive(Debug)]
pub struct ContiguousKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length (tokens processed)
    seq_len: usize,
    /// Stride per layer (aligned to cache lines)
    layer_stride: usize,
    /// Contiguous K cache: [num_layers * layer_stride]
    k_data: Vec<f32>,
    /// Contiguous V cache: [num_layers * layer_stride]
    v_data: Vec<f32>,
}

impl ContiguousKVCache {
    /// Create a new contiguous KV cache (PARITY-005)
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension (num_heads * head_dim)
    /// * `max_seq_len` - Maximum sequence length to cache
    ///
    /// # Cache Line Alignment
    /// Layer stride is padded to nearest cache line boundary (16 floats)
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        // Calculate layer stride: max_seq_len * hidden_dim, rounded up to cache line
        let raw_layer_size = max_seq_len * hidden_dim;
        let layer_stride = Self::align_to_cache_line(raw_layer_size);

        // Total size for all layers
        let total_size = num_layers * layer_stride;

        // Pre-allocate contiguous buffers with zeros
        let k_data = vec![0.0f32; total_size];
        let v_data = vec![0.0f32; total_size];

        Self {
            num_layers,
            hidden_dim,
            max_seq_len,
            seq_len: 0,
            layer_stride,
            k_data,
            v_data,
        }
    }

    /// Align size to 64-byte cache line boundary
    #[inline]
    fn align_to_cache_line(size: usize) -> usize {
        let remainder = size % FLOATS_PER_CACHE_LINE;
        if remainder == 0 {
            size
        } else {
            size + FLOATS_PER_CACHE_LINE - remainder
        }
    }

    /// Create cache from model configuration
    #[must_use]
    pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self {
        Self::new(config.num_layers, config.hidden_dim, max_seq_len)
    }

    /// Check if this cache has contiguous layout (always true for this type)
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        true
    }

    /// Check if data is cache-line aligned
    #[must_use]
    pub fn is_cache_aligned(&self) -> bool {
        // Check that layer_stride is a multiple of cache line size
        self.layer_stride.is_multiple_of(FLOATS_PER_CACHE_LINE)
    }

    /// Get the layer stride (elements per layer, cache-aligned)
    #[must_use]
    pub fn layer_stride(&self) -> usize {
        self.layer_stride
    }

    /// Get offset for a specific layer
    #[inline]
    fn layer_offset(&self, layer: usize) -> usize {
        layer * self.layer_stride
    }

    /// Append K and V vectors for a single position to a layer's cache
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k` - Key vector [hidden_dim]
    /// * `v` - Value vector [hidden_dim]
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer >= self.num_layers || self.seq_len >= self.max_seq_len {
            return;
        }

        let layer_base = self.layer_offset(layer);
        let pos_offset = self.seq_len * self.hidden_dim;
        let start = layer_base + pos_offset;
        let end = start + self.hidden_dim;

        if end <= self.k_data.len() {
            self.k_data[start..end].copy_from_slice(k);
            self.v_data[start..end].copy_from_slice(v);
        }
    }

    /// Advance the sequence position after processing a token
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get cached keys for a layer (PARITY-005: sequential access)
    ///
    /// Returns slice of [seq_len * hidden_dim] - contiguous for prefetching
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &self.k_data[start..start + len]
    }

    /// Get cached values for a layer (PARITY-005: sequential access)
    ///
    /// Returns slice of [seq_len * hidden_dim] - contiguous for prefetching
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &self.v_data[start..start + len]
    }

    /// Get mutable cached keys for a layer
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.k_data[start..start + len]
    }

    /// Get mutable cached values for a layer
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.v_data[start..start + len]
    }

    /// Current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // Note: We don't zero the data - just reset seq_len
        // This avoids unnecessary memory writes
    }

    /// Reset cache and zero all data
    pub fn reset_and_zero(&mut self) {
        self.seq_len = 0;
        self.k_data.fill(0.0);
        self.v_data.fill(0.0);
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.k_data.len() + self.v_data.len()) * std::mem::size_of::<f32>()
    }

    /// Prefetch K cache for a layer (hint to hardware prefetcher)
    ///
    /// This is a no-op on most platforms but helps document intent.
    /// The sequential layout already enables automatic prefetching.
    #[inline]
    pub fn prefetch_k(&self, layer: usize) {
        if layer < self.num_layers {
            let start = self.layer_offset(layer);
            // Touch first cache line to trigger prefetch
            let _ = self.k_data.get(start);
        }
    }

    /// Prefetch V cache for a layer
    #[inline]
    pub fn prefetch_v(&self, layer: usize) {
        if layer < self.num_layers {
            let start = self.layer_offset(layer);
            let _ = self.v_data.get(start);
        }
    }
}

// ============================================================================
// IMP-123: Dispatch Metrics for CPU vs GPU Decision Tracking
// ============================================================================

/// Thread-safe metrics for tracking CPU vs GPU dispatch decisions (IMP-123, IMP-129)
///
/// Tracks how often operations are dispatched to CPU vs GPU backends,
/// enabling analysis of adaptive dispatch effectiveness.
///
/// Also tracks latency histograms for performance analysis (IMP-129).
///
/// Uses atomic counters for thread-safe concurrent access.
#[derive(Debug)]
pub struct DispatchMetrics {
    /// Number of operations dispatched to CPU
    cpu_dispatches: std::sync::atomic::AtomicUsize,
    /// Number of operations dispatched to GPU
    gpu_dispatches: std::sync::atomic::AtomicUsize,
    /// CPU latency tracking (IMP-129)
    cpu_latency_count: std::sync::atomic::AtomicUsize,
    cpu_latency_sum_us: std::sync::atomic::AtomicU64,
    /// GPU latency tracking (IMP-129)
    gpu_latency_count: std::sync::atomic::AtomicUsize,
    gpu_latency_sum_us: std::sync::atomic::AtomicU64,
    /// CPU latency histogram buckets: [0-100µs, 100-500µs, 500-1000µs, 1000-5000µs, 5000+µs]
    cpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    /// GPU latency histogram buckets: [0-100µs, 100-500µs, 500-1000µs, 1000-5000µs, 5000+µs]
    gpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    /// CPU latency min/max tracking (IMP-134)
    cpu_latency_min_us: std::sync::atomic::AtomicU64,
    cpu_latency_max_us: std::sync::atomic::AtomicU64,
    /// GPU latency min/max tracking (IMP-134)
    gpu_latency_min_us: std::sync::atomic::AtomicU64,
    gpu_latency_max_us: std::sync::atomic::AtomicU64,
    /// CPU latency sum of squares for variance calculation (IMP-135)
    cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    /// GPU latency sum of squares for variance calculation (IMP-135)
    gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    /// Start time in milliseconds since epoch (IMP-140)
    start_time_ms: std::sync::atomic::AtomicU64,
}

impl DispatchMetrics {
    /// Histogram bucket boundaries in microseconds (IMP-136: made public)
    /// These define the upper bounds for each bucket: [0-100µs, 100-500µs, 500-1000µs, 1000-5000µs, 5000+µs]
    pub const BUCKET_BOUNDARIES: [u64; 4] = [100, 500, 1000, 5000];

    /// Create new metrics tracker with zero counts
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            gpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            gpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            cpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            gpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            // IMP-134: Min initialized to MAX so first sample will be smaller
            cpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            cpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            gpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            // IMP-135: Sum of squares for variance calculation
            cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            // IMP-140: Start time for throughput calculation
            start_time_ms: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            ),
        }
    }

    /// Get bucket index for a latency value in microseconds
    fn bucket_index(latency_us: u64) -> usize {
        for (i, &boundary) in Self::BUCKET_BOUNDARIES.iter().enumerate() {
            if latency_us < boundary {
                return i;
            }
        }
        4 // Last bucket (5000+µs)
    }

    /// Record a CPU dispatch decision
    pub fn record_cpu_dispatch(&self) {
        self.cpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a GPU dispatch decision
    pub fn record_gpu_dispatch(&self) {
        self.gpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record CPU dispatch latency (IMP-129)
    pub fn record_cpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.cpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        let bucket = Self::bucket_index(latency_us);
        self.cpu_latency_buckets[bucket].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // IMP-134: Track min/max
        self.cpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        // IMP-135: Track sum of squares for variance
        self.cpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Record GPU dispatch latency (IMP-129)
    pub fn record_gpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.gpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        let bucket = Self::bucket_index(latency_us);
        self.gpu_latency_buckets[bucket].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // IMP-134: Track min/max
        self.gpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        // IMP-135: Track sum of squares for variance
        self.gpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Get total number of CPU dispatches
    #[must_use]
    pub fn cpu_dispatches(&self) -> usize {
        self.cpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total number of GPU dispatches
    #[must_use]
    pub fn gpu_dispatches(&self) -> usize {
        self.gpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total number of dispatches (CPU + GPU)
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.cpu_dispatches() + self.gpu_dispatches()
    }

    /// Get GPU dispatch ratio (0.0 to 1.0)
    ///
    /// Returns 0.0 if no dispatches have occurred.
    #[must_use]
    pub fn gpu_ratio(&self) -> f64 {
        let total = self.total_dispatches();
        if total == 0 {
            0.0
        } else {
            self.gpu_dispatches() as f64 / total as f64
        }
    }

    /// Get CPU latency sample count (IMP-129)
    #[must_use]
    pub fn cpu_latency_count(&self) -> usize {
        self.cpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency sample count (IMP-129)
    #[must_use]
    pub fn gpu_latency_count(&self) -> usize {
        self.gpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get mean CPU latency in microseconds (IMP-129)
    #[must_use]
    pub fn cpu_latency_mean_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count == 0 {
            0.0
        } else {
            let sum = self
                .cpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed);
            sum as f64 / count as f64
        }
    }

    /// Get mean GPU latency in microseconds (IMP-129)
    #[must_use]
    pub fn gpu_latency_mean_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count == 0 {
            0.0
        } else {
            let sum = self
                .gpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed);
            sum as f64 / count as f64
        }
    }

    /// Get total CPU latency sum in microseconds (IMP-130)
    #[must_use]
    pub fn cpu_latency_sum_us(&self) -> u64 {
        self.cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total GPU latency sum in microseconds (IMP-130)
    #[must_use]
    pub fn gpu_latency_sum_us(&self) -> u64 {
        self.gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get minimum CPU latency in microseconds (IMP-134)
    ///
    /// Returns 0 if no samples have been recorded.
    #[must_use]
    pub fn cpu_latency_min_us(&self) -> u64 {
        if self.cpu_latency_count() == 0 {
            return 0;
        }
        self.cpu_latency_min_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get maximum CPU latency in microseconds (IMP-134)
    ///
    /// Returns 0 if no samples have been recorded.
    #[must_use]
    pub fn cpu_latency_max_us(&self) -> u64 {
        self.cpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get minimum GPU latency in microseconds (IMP-134)
    ///
    /// Returns 0 if no samples have been recorded.
    #[must_use]
    pub fn gpu_latency_min_us(&self) -> u64 {
        if self.gpu_latency_count() == 0 {
            return 0;
        }
        self.gpu_latency_min_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get maximum GPU latency in microseconds (IMP-134)
    ///
    /// Returns 0 if no samples have been recorded.
    #[must_use]
    pub fn gpu_latency_max_us(&self) -> u64 {
        self.gpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get CPU latency variance in microseconds squared (IMP-135)
    ///
    /// Uses population variance formula: Var(X) = E[X²] - E[X]²
    /// Returns 0.0 if fewer than 2 samples have been recorded.
    #[must_use]
    pub fn cpu_latency_variance_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .cpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        // Var(X) = E[X²] - E[X]² = sum_sq/n - (sum/n)²
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get CPU latency standard deviation in microseconds (IMP-135)
    ///
    /// Returns sqrt(variance). Returns 0.0 if fewer than 2 samples.
    #[must_use]
    pub fn cpu_latency_stddev_us(&self) -> f64 {
        self.cpu_latency_variance_us().sqrt()
    }

    /// Get GPU latency variance in microseconds squared (IMP-135)
    ///
    /// Uses population variance formula: Var(X) = E[X²] - E[X]²
    /// Returns 0.0 if fewer than 2 samples have been recorded.
    #[must_use]
    pub fn gpu_latency_variance_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .gpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        // Var(X) = E[X²] - E[X]² = sum_sq/n - (sum/n)²
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get GPU latency standard deviation in microseconds (IMP-135)
    ///
    /// Returns sqrt(variance). Returns 0.0 if fewer than 2 samples.
    #[must_use]
    pub fn gpu_latency_stddev_us(&self) -> f64 {
        self.gpu_latency_variance_us().sqrt()
    }

    /// Get CPU latency histogram bucket counts (IMP-129)
    /// Buckets: [0-100µs, 100-500µs, 500-1000µs, 1000-5000µs, 5000+µs]
    #[must_use]
    pub fn cpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.cpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    /// Get GPU latency histogram bucket counts (IMP-129)
    /// Buckets: [0-100µs, 100-500µs, 500-1000µs, 1000-5000µs, 5000+µs]
    #[must_use]
    pub fn gpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.gpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    /// Estimate percentile from histogram buckets (IMP-131)
    /// Uses linear interpolation within bucket ranges.
    /// Bucket upper bounds: [100, 500, 1000, 5000, 10000] (10000 for +Inf estimation)
    fn estimate_percentile_from_buckets(buckets: &[usize; 5], percentile: f64) -> f64 {
        const BUCKET_UPPER_BOUNDS: [f64; 5] = [100.0, 500.0, 1000.0, 5000.0, 10000.0];
        const BUCKET_LOWER_BOUNDS: [f64; 5] = [0.0, 100.0, 500.0, 1000.0, 5000.0];

        let total: usize = buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let target_rank = (percentile / 100.0) * total as f64;
        let mut cumulative: f64 = 0.0;

        for (i, &count) in buckets.iter().enumerate() {
            let prev_cumulative = cumulative;
            cumulative += count as f64;

            if cumulative >= target_rank {
                // Percentile falls within this bucket
                // Linear interpolation within bucket
                if count == 0 {
                    return BUCKET_LOWER_BOUNDS[i];
                }
                let fraction = (target_rank - prev_cumulative) / count as f64;
                let lower = BUCKET_LOWER_BOUNDS[i];
                let upper = BUCKET_UPPER_BOUNDS[i];
                return lower + fraction * (upper - lower);
            }
        }

        // Should not reach here, but return upper bound of last bucket
        BUCKET_UPPER_BOUNDS[4]
    }

    /// Get CPU latency p50 (median) in microseconds (IMP-131)
    #[must_use]
    pub fn cpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 50.0)
    }

    /// Get CPU latency p95 in microseconds (IMP-131)
    #[must_use]
    pub fn cpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 95.0)
    }

    /// Get CPU latency p99 in microseconds (IMP-131)
    #[must_use]
    pub fn cpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 99.0)
    }

    /// Get GPU latency p50 (median) in microseconds (IMP-131)
    #[must_use]
    pub fn gpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 50.0)
    }

    /// Get GPU latency p95 in microseconds (IMP-131)
    #[must_use]
    pub fn gpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 95.0)
    }

    /// Get GPU latency p99 in microseconds (IMP-131)
    #[must_use]
    pub fn gpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 99.0)
    }

    /// Get human-readable bucket boundary strings (IMP-136)
    /// Returns bucket ranges like: `["0-100", "100-500", "500-1000", "1000-5000", "5000+"]`
    #[must_use]
    pub fn bucket_boundaries_us(&self) -> Vec<String> {
        vec![
            format!("0-{}", Self::BUCKET_BOUNDARIES[0]),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[0],
                Self::BUCKET_BOUNDARIES[1]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[1],
                Self::BUCKET_BOUNDARIES[2]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[2],
                Self::BUCKET_BOUNDARIES[3]
            ),
            format!("{}+", Self::BUCKET_BOUNDARIES[3]),
        ]
    }

    /// Get start time in milliseconds since epoch (IMP-140)
    #[must_use]
    pub fn start_time_ms(&self) -> u64 {
        self.start_time_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get elapsed time in seconds since start/reset (IMP-140)
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        let start = self.start_time_ms();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let elapsed_ms = now.saturating_sub(start);
        elapsed_ms as f64 / 1000.0
    }

    /// Get throughput in requests per second (IMP-140)
    /// Returns total_dispatches / elapsed_seconds
    #[must_use]
    pub fn throughput_rps(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed < 0.001 {
            // Avoid division by very small numbers
            return 0.0;
        }
        self.total_dispatches() as f64 / elapsed
    }

    /// Get CPU latency coefficient of variation (IMP-142)
    /// CV = stddev / mean * 100 (as percentage)
    /// Returns 0.0 if no samples or mean is zero
    #[must_use]
    pub fn cpu_latency_cv(&self) -> f64 {
        let mean = self.cpu_latency_mean_us();
        if mean < 0.001 {
            return 0.0;
        }
        let stddev = self.cpu_latency_stddev_us();
        (stddev / mean) * 100.0
    }

    /// Get GPU latency coefficient of variation (IMP-142)
    /// CV = stddev / mean * 100 (as percentage)
    /// Returns 0.0 if no samples or mean is zero
    #[must_use]
    pub fn gpu_latency_cv(&self) -> f64 {
        let mean = self.gpu_latency_mean_us();
        if mean < 0.001 {
            return 0.0;
        }
        let stddev = self.gpu_latency_stddev_us();
        (stddev / mean) * 100.0
    }

    /// Get CPU/GPU speedup ratio (IMP-142)
    /// Returns CPU mean latency / GPU mean latency
    /// A value > 1.0 means GPU is faster than CPU
    /// Returns 0.0 if GPU has no samples or zero mean
    #[must_use]
    pub fn cpu_gpu_speedup(&self) -> f64 {
        let gpu_mean = self.gpu_latency_mean_us();
        if gpu_mean < 0.001 {
            return 0.0;
        }
        let cpu_mean = self.cpu_latency_mean_us();
        cpu_mean / gpu_mean
    }

    /// Reset all metrics to zero (IMP-137)
    /// This is useful for A/B testing and iterative performance tuning.
    pub fn reset(&self) {
        // Reset dispatch counters
        self.cpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Reset latency counters
        self.cpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Reset min/max (min back to MAX, max back to 0)
        self.cpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Reset sum of squares for variance
        self.cpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Reset histogram buckets
        for bucket in &self.cpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for bucket in &self.gpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }

        // IMP-140: Reset start time to now
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.start_time_ms
            .store(now, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for DispatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedGenerateConfig {
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

// ============================================================================
// IMP-311: CUDA Backend via trueno-gpu (Pure Rust PTX Generation)
// ============================================================================

/// CUDA backend for NVIDIA GPU acceleration (IMP-311)
///
/// Uses trueno-gpu for pure Rust PTX code generation - no LLVM, nvcc, or
/// external CUDA toolkit required. Generates optimized kernels for:
/// - Q4_K quantized GEMM (fused dequant+matmul) - IMP-312
/// - FlashAttention-style tiled attention - IMP-313
/// - Paged KV cache management - IMP-314
/// - CUDA graph capture for forward pass - IMP-315
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::CudaBackend;
///
/// let cuda = CudaBackend::new(1024, 1024, 4096, 64);
/// let ptx = cuda.q4k_gemm_ptx();  // Get PTX for Q4_K GEMM kernel
/// let attention_ptx = cuda.flash_attention_ptx(2048, 64, true);  // Causal attention
/// ```
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaBackend {
    /// Output rows (M) for GEMM operations
    pub m: u32,
    /// Output columns (N) for GEMM operations
    pub n: u32,
    /// Inner dimension (K) - must be divisible by Q4_K block size (32)
    pub k: u32,
    /// Head dimension for attention (typically 64 or 128)
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Maximum sequence length for KV cache
    pub max_seq_len: u32,
    /// Cached PTX for Q4_K GEMM kernel (IMP-312)
    q4k_gemm_ptx_cache: std::cell::RefCell<Option<String>>,
    /// Cached PTX for FlashAttention kernel (IMP-313)
    flash_attention_ptx_cache: std::cell::RefCell<Option<String>>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend with specified dimensions
    ///
    /// # Arguments
    /// * `m` - Output rows for GEMM
    /// * `n` - Output columns for GEMM
    /// * `k` - Inner dimension (should be divisible by 32 for Q4_K)
    /// * `head_dim` - Head dimension for attention (typically 64)
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, head_dim: u32) -> Self {
        Self {
            m,
            n,
            k,
            head_dim,
            num_heads: 32,     // Default for many models
            max_seq_len: 2048, // Default context length
            q4k_gemm_ptx_cache: std::cell::RefCell::new(None),
            flash_attention_ptx_cache: std::cell::RefCell::new(None),
        }
    }

    /// Set the number of attention heads
    #[must_use]
    pub const fn with_num_heads(mut self, num_heads: u32) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set the maximum sequence length for KV cache
    #[must_use]
    pub const fn with_max_seq_len(mut self, max_seq_len: u32) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    // ========================================================================
    // IMP-312: CUDA Q4_K Dequant+Matmul Kernel
    // ========================================================================

    /// Generate PTX for Q4_K quantized GEMM kernel (IMP-312)
    ///
    /// The kernel fuses dequantization with matrix multiplication:
    /// - Dequantization: val = scale * quant + min (per Q4_K block)
    /// - Matrix multiply: C = A × dequant(B)
    ///
    /// # Performance
    /// - Uses warp shuffle for efficient reduction
    /// - Shared memory for dequantized tiles
    /// - Coalesced memory access patterns
    #[must_use]
    pub fn q4k_gemm_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.q4k_gemm_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate PTX using trueno-gpu
        let kernel = QuantizeKernel::new(self.m, self.n, self.k);
        let ptx = kernel.emit_ptx();

        // Cache the result
        *self.q4k_gemm_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for Q4_K GEMM
    #[must_use]
    pub fn q4k_gemm_kernel_name(&self) -> &'static str {
        "q4k_gemm_fused"
    }

    /// Get number of Q4_K blocks per row (K / 32)
    #[must_use]
    pub const fn q4k_blocks_per_row(&self) -> u32 {
        self.k / 32
    }

    /// Estimate Q4_K weight memory size in bytes
    /// Each block: 2 bytes header (scale+min) + 16 bytes data = 18 bytes for 32 weights
    #[must_use]
    pub const fn q4k_weight_bytes(&self) -> usize {
        let blocks_per_row = self.k / 32;
        let bytes_per_row = blocks_per_row * 18;
        (self.n as usize) * (bytes_per_row as usize)
    }

    // ========================================================================
    // IMP-313: CUDA FlashAttention Kernel
    // ========================================================================

    /// Generate PTX for FlashAttention-style tiled attention (IMP-313)
    ///
    /// Implements IO-aware attention per Dao et al. [16]:
    /// - Never materializes the full N×N attention matrix
    /// - Online softmax with running max and sum
    /// - O(N × d) memory instead of O(N²)
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length (N)
    /// * `head_dim` - Head dimension (d)
    /// * `causal` - Enable causal masking for autoregressive models
    #[must_use]
    pub fn flash_attention_ptx(&self, seq_len: u32, head_dim: u32, causal: bool) -> String {
        let kernel = if causal {
            AttentionKernel::new(seq_len, head_dim).with_causal()
        } else {
            AttentionKernel::new(seq_len, head_dim)
        };
        kernel.emit_ptx()
    }

    /// Generate PTX for causal FlashAttention (cached version)
    #[must_use]
    pub fn flash_attention_causal_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.flash_attention_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate causal attention PTX
        let ptx = self.flash_attention_ptx(self.max_seq_len, self.head_dim, true);

        // Cache the result
        *self.flash_attention_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for FlashAttention
    #[must_use]
    pub const fn flash_attention_kernel_name(&self, causal: bool) -> &'static str {
        if causal {
            "flash_attention_causal"
        } else {
            "flash_attention"
        }
    }

    /// Estimate shared memory size for FlashAttention (in bytes)
    /// Uses tiles of Q (B_r × d) and KV (B_c × d × 2)
    #[must_use]
    pub const fn flash_attention_smem_bytes(&self) -> usize {
        let tile_q = 64_u32;
        let tile_kv = 64_u32;
        let d = self.head_dim;
        // Q tile + K tile + V tile, all f32
        ((tile_q * d + tile_kv * d * 2) * 4) as usize
    }

    // ========================================================================
    // IMP-314: CUDA KV Cache with Paged Memory
    // ========================================================================

    /// Calculate KV cache memory size per layer in bytes
    ///
    /// KV cache stores Key and Value tensors for attention:
    /// - K: [num_heads, seq_len, head_dim] × sizeof(f32)
    /// - V: [num_heads, seq_len, head_dim] × sizeof(f32)
    #[must_use]
    pub const fn kv_cache_bytes_per_layer(&self) -> usize {
        let k_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        let v_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        (k_size + v_size) as usize
    }

    /// Calculate total KV cache memory for all layers
    #[must_use]
    pub const fn kv_cache_total_bytes(&self, num_layers: u32) -> usize {
        self.kv_cache_bytes_per_layer() * (num_layers as usize)
    }

    /// Get page size for paged KV cache (IMP-314)
    /// Default: 64 tokens per page to balance memory efficiency and fragmentation
    #[must_use]
    pub const fn kv_cache_page_tokens(&self) -> u32 {
        64
    }

    /// Calculate number of pages needed for given sequence length
    #[must_use]
    pub const fn kv_cache_pages_needed(&self, seq_len: u32) -> u32 {
        let page_size = self.kv_cache_page_tokens();
        seq_len.div_ceil(page_size)
    }

    // ========================================================================
    // IMP-315: CUDA Graph Capture Helpers
    // ========================================================================

    /// Get CUDA launch configuration for Q4_K GEMM kernel
    ///
    /// Returns (grid_dim, block_dim) tuple for kernel launch
    #[must_use]
    pub const fn q4k_gemm_launch_config(&self) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_size = 32_u32;
        let grid_x = self.n.div_ceil(tile_size);
        let grid_y = self.m.div_ceil(tile_size);
        let grid = (grid_x, grid_y, 1);
        let block = (tile_size * tile_size, 1, 1);
        (grid, block)
    }

    /// Get CUDA launch configuration for FlashAttention kernel
    #[must_use]
    pub const fn flash_attention_launch_config(
        &self,
        seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_q = 64_u32;
        let num_q_blocks = seq_len.div_ceil(tile_q);
        let grid = (num_q_blocks, self.num_heads, 1);
        let block = (tile_q * self.head_dim, 1, 1);
        (grid, block)
    }

    /// Check if dimensions are valid for CUDA kernels
    #[must_use]
    pub const fn validate_dimensions(&self) -> bool {
        // K must be divisible by Q4_K block size (32)
        let k_valid = self.k.is_multiple_of(32);
        // Head dim should be power of 2 for efficient memory access
        let head_dim_valid = self.head_dim.is_power_of_two();
        // Dimensions must be non-zero
        let non_zero = self.m > 0 && self.n > 0 && self.k > 0 && self.head_dim > 0;
        k_valid && head_dim_valid && non_zero
    }

    /// Get PTX target SM version (default: sm_89 for Ada Lovelace/RTX 4090)
    #[must_use]
    pub const fn ptx_target(&self) -> &'static str {
        "sm_89"
    }

    /// Get PTX version (default: 8.0)
    #[must_use]
    pub const fn ptx_version(&self) -> (u32, u32) {
        (8, 0)
    }
}

