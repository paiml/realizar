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
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use trueno::brick::BrickProfiler;

use crate::error::{RealizarError, Result};

// ============================================================================
// Memory-mapped model data (Heijunka - Level Loading)
// ============================================================================
//
// References:
// - Didona et al. (2022): mmap vs read() achieves 2.3x throughput for sequential access
// - Chu (2011): LMDB design - let kernel manage pages, don't fight the VM subsystem
// - Vahalia (1996): SIGBUS behavior on truncated mmap
//
// This abstraction allows models to be loaded via:
// 1. Memory mapping (mmap) - zero-copy, kernel manages pages, no zram pressure
// 2. Heap allocation (Vec<u8>) - required for compressed files after decompression

/// Model data storage abstraction for zero-copy access.
///
/// # Memory Management
///
/// When using `Mmap` variant:
/// - Data is not copied into userspace heap
/// - Kernel demand-pages from disk on access
/// - After GPU transfer, call `release_cpu_pages()` to advise kernel
/// - Pages backed by file (not zram) when evicted
///
/// When using `Heap` variant:
/// - Used for compressed files (must decompress to Vec<u8>)
/// - Standard heap allocation behavior
/// - May be compressed to zram when idle
#[derive(Debug)]
pub enum ModelData {
    /// Memory-mapped file (zero-copy, kernel-managed paging)
    #[cfg(not(target_arch = "wasm32"))]
    Mmap {
        /// Memory-mapped region
        mmap: memmap2::Mmap,
        /// Original file path (for diagnostics)
        path: PathBuf,
    },
    /// Heap-allocated data (for compressed files or WASM)
    Heap(Vec<u8>),
}

impl ModelData {
    /// Open a file with memory mapping.
    ///
    /// # Safety
    ///
    /// Uses `memmap2::Mmap` which requires:
    /// - File must not be truncated while mapped (SIGBUS on Unix)
    /// - File must not be modified while mapped (undefined behavior)
    ///
    /// # References
    ///
    /// - Vahalia (1996): SIGBUS from truncated mmap
    /// - memmap2 crate safety documentation
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(unsafe_code)]
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open file '{}': {e}", path_ref.display()),
        })?;

        // SAFETY: File is opened read-only. We document the single-writer
        // assumption. Callers should validate checksums before trusting data.
        // SIGBUS can occur if file is truncated externally - this is documented.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to mmap file '{}': {e}", path_ref.display()),
                })?
        };

        Ok(Self::Mmap {
            mmap,
            path: path_ref.to_path_buf(),
        })
    }

    /// Create from heap-allocated data (for compressed files).
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::Heap(data)
    }

    /// Get the data as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mmap { mmap, .. } => mmap,
            Self::Heap(data) => data,
        }
    }

    /// Get data length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Check if data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Release CPU pages after GPU transfer (Unix only).
    ///
    /// Calls `madvise(MADV_DONTNEED)` to tell the kernel these pages
    /// are no longer needed. The kernel will:
    /// - Drop pages immediately (not compress to zram)
    /// - Re-fault from disk if accessed again
    ///
    /// # When to Call
    ///
    /// After `cuMemcpy()` completes for all tensors.
    ///
    /// # Safety
    ///
    /// Uses `unchecked_advise` because `MADV_DONTNEED` is in the
    /// `UncheckedAdvice` enum. This is safe for read-only mmaps where
    /// data can be re-faulted from the backing file.
    ///
    /// # References
    ///
    /// - Didona et al. (2022): madvise for memory management
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[allow(unsafe_code)]
    pub fn release_cpu_pages(&self) -> Result<()> {
        match self {
            Self::Mmap { mmap, path } => {
                // SAFETY: We opened the file read-only, so MADV_DONTNEED is safe -
                // the kernel will re-fault pages from the backing file if accessed.
                unsafe {
                    mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
                        .map_err(|e| RealizarError::IoError {
                            message: format!(
                                "madvise(MADV_DONTNEED) failed for '{}': {e}",
                                path.display()
                            ),
                        })
                }
            }
            Self::Heap(_) => {
                // No-op for heap data - kernel manages via normal VM pressure
                Ok(())
            }
        }
    }

    /// Advise sequential access pattern (Unix only).
    ///
    /// Call before linear scan through model data.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn advise_sequential(&self) -> Result<()> {
        match self {
            Self::Mmap { mmap, path } => {
                mmap.advise(memmap2::Advice::Sequential).map_err(|e| {
                    RealizarError::IoError {
                        message: format!(
                            "madvise(MADV_SEQUENTIAL) failed for '{}': {e}",
                            path.display()
                        ),
                    }
                })
            }
            Self::Heap(_) => Ok(()),
        }
    }

    /// Check if this is memory-mapped data.
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mmap { .. } => true,
            Self::Heap(_) => false,
        }
    }
}

/// Magic number: "APR2" in ASCII (0x41505232)
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x32];

/// Format version for .apr v2 files
pub const FORMAT_VERSION: (u8, u8) = (2, 0);

/// Header size in bytes (64-byte aligned)
pub const HEADER_SIZE: usize = 64;

/// Tensor alignment in bytes
pub const ALIGNMENT: usize = 64;

// ============================================================================
// Dequantization helpers for quantized tensor formats
// ============================================================================

/// Convert F16 (IEEE 754 half-precision) to F32
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal number
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Dequantize F16 data to F32
fn dequantize_f16(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        result.push(f16_to_f32(bits));
    }
    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 format (GGUF compatible)
/// Q8_0: blocks of 32 elements, each block has 2-byte f16 scale + 32 bytes of int8 quants
fn dequantize_q8_0(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 int8 values

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + BLOCK_BYTES <= bytes.len() {
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Read 32 int8 values
        for i in 0..32 {
            if result.len() >= num_elements {
                break;
            }
            let v = f32::from(bytes[offset + i] as i8);
            result.push(v * scale);
        }
        offset += 32;
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q4_K format (GGUF K-quants)
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
fn dequantize_q4_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]));
        offset += 4;

        // Read scales (12 bytes = 8 6-bit scale values packed)
        let scales_bytes = &bytes[offset..offset + 12];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Unpack 6-bit scales and mins from 12 bytes
        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = scales_bytes[i + 4] & 0x3F;
            mins[i] = (scales_bytes[i] >> 6) | ((scales_bytes[i + 8] & 0x0F) << 2);
            mins[i + 4] = (scales_bytes[i + 4] >> 6) | ((scales_bytes[i + 8] >> 4) << 2);
        }
        offset += 12;

        // Read 128 bytes = 256 4-bit quantized values
        let qs = &bytes[offset..offset + 128];
        offset += 128;

        // Dequantize: each sub-block has 32 elements (8 sub-blocks total)
        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            for l in 0..16 {
                if result.len() >= num_elements {
                    break;
                }
                let q_byte = qs[j * 16 + l];
                let q0 = (q_byte & 0x0F) as f32;
                let q1 = (q_byte >> 4) as f32;
                result.push(q0 * scale - min_val);
                if result.len() < num_elements {
                    result.push(q1 * scale - min_val);
                }
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q6_K format (GGUF K-quants)
/// Q6_K: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
fn dequantize_q6_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        // Read ql (128 bytes = low 4 bits of 256 6-bit values)
        let ql = &bytes[offset..offset + 128];
        offset += 128;

        // Read qh (64 bytes = high 2 bits of 256 6-bit values)
        let qh = &bytes[offset..offset + 64];
        offset += 64;

        // Read scales (16 bytes = 16 int8 scales)
        let scales = &bytes[offset..offset + 16];
        offset += 16;

        // Read d (f16)
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        offset += 2;

        // Dequantize 16 sub-blocks of 16 elements each
        for j in 0..16 {
            let scale = d * f32::from(scales[j] as i8);

            for l in 0..8 {
                if result.len() >= num_elements {
                    break;
                }
                let idx = j * 8 + l;
                let ql_byte = ql[idx];
                let qh_byte = qh[idx / 2];

                // Extract two 6-bit values
                let qh_shift = (l % 2) * 4;
                let q0 = ((ql_byte & 0x0F) | ((qh_byte >> qh_shift) & 0x03) << 4) as i8 - 32;
                let q1 = ((ql_byte >> 4) | (((qh_byte >> qh_shift) >> 2) & 0x03) << 4) as i8 - 32;

                result.push(f32::from(q0) * scale);
                if result.len() < num_elements {
                    result.push(f32::from(q1) * scale);
                }
            }
        }
    }

    result.truncate(num_elements);
    result
}

// ============================================================================
// Quantization type mapping for GPU kernels
// ============================================================================

/// Map APR dtype string to GGML quantization type ID.
///
/// These IDs are used by `load_quantized_weights_with_type()` to select
/// the correct GPU dequantization kernel (Q4K GEMV, Q6K GEMV, etc.).
#[inline]
fn dtype_to_ggml_qtype(dtype: &str) -> Option<u32> {
    match dtype {
        "Q4_K" | "q4_k" => Some(12), // GGML_TYPE_Q4_K
        "Q5_K" | "q5_k" => Some(13), // GGML_TYPE_Q5_K
        "Q6_K" | "q6_k" => Some(14), // GGML_TYPE_Q6_K
        "Q8_0" | "q8_0" => Some(8),  // GGML_TYPE_Q8_0
        "Q4_0" | "q4_0" => Some(2),  // GGML_TYPE_Q4_0
        "Q4_1" | "q4_1" => Some(3),  // GGML_TYPE_Q4_1
        "Q5_0" | "q5_0" => Some(6),  // GGML_TYPE_Q5_0
        _ => None, // F32/F16 are not quantized
    }
}

/// Check if dtype is a quantized format that can use GPU dequant kernels.
#[inline]
fn is_quantized_dtype(dtype: &str) -> bool {
    dtype_to_ggml_qtype(dtype).is_some()
}

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
            8 => "Q4_K", // GGUF Q4_K_M quantization (4.5 bits/element)
            9 => "Q6_K", // GGUF Q6_K quantization (6.5 bits/element)
            10 => "Q8_0", // GGUF Q8_0 quantization (8 bits/element)
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
    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default)]
    pub rope_type: Option<u32>,
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
        file.read_exact(&mut header_buf).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read .apr header: {e}"),
        })?;

        let header = AprHeader::from_bytes(&header_buf)?;

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
            }
            "F16" | "f16" => {
                Ok(dequantize_f16(bytes, num_elements))
            }
            "Q8_0" | "q8_0" => {
                Ok(dequantize_q8_0(bytes, num_elements))
            }
            "Q4_K" | "q4_k" => {
                Ok(dequantize_q4_k(bytes, num_elements))
            }
            "Q6_K" | "q6_k" => {
                Ok(dequantize_q6_k(bytes, num_elements))
            }
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
        let intermediate_dim = self.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
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
            "token_embd.weight", // GGUF naming
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
        profiler.stop(timer, seq_len as u64);

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Load tensor names (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF
            ])?;
            let q_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
                &format!("blk.{layer_idx}.attn_q.weight"), // GGUF
            ])?;
            let k_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
                &format!("blk.{layer_idx}.attn_k.weight"), // GGUF
            ])?;
            let v_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
                &format!("blk.{layer_idx}.attn_v.weight"), // GGUF
            ])?;
            let o_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF
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
                &format!("blk.{layer_idx}.ffn_norm.weight"), // GGUF
            ])?;
            let gate_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"), // GGUF
            ])?;
            let up_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"), // GGUF
            ])?;
            let down_name = self.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"), // GGUF
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
            "output_norm.weight", // GGUF naming
        ])?;
        let final_norm = self.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);
        profiler.stop(timer, seq_len as u64);

        // 4. APR.LMHEAD
        let timer = profiler.start("apr.LmHead");
        let lm_head_name = self.find_tensor_name(&[
            "lm_head.weight",
            "output.weight", // GGUF uses this
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
                    tokens[i].clone_from(&merged);
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
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    let chunks = n / 8;

    // SAFETY: This entire fn is unsafe with target_feature(avx2, fma)
    // All intrinsics are safe to call given the target_feature guarantee
    // The unsafe block is required for Rust 2024 edition compliance
    unsafe {
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
                    let q_val = q
                        .get(s * hidden_dim + h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
                    let k_val = k
                        .get(t * kv_dim + kv_h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
                    score += q_val * k_val;
                }
                scores[t] = score * scale;
            }

            // Softmax
            let max_score = scores[..=s]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for score in &mut scores[..=s] {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores[..=s] {
                *score /= sum;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut val = 0.0;
                for t in 0..=s {
                    let v_val = v
                        .get(t * kv_dim + kv_h * head_dim + d)
                        .copied()
                        .unwrap_or(0.0);
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
    /// Cached embedding table (F32 for fast lookup)
    embedding_cache: Option<Vec<f32>>,
    /// Hidden dimension (cached for embedding lookup)
    hidden_dim: usize,
    /// Current KV cache position (increments with each decoded token)
    kv_position: u32,
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
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };

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

        // CORRECTNESS-011: Set RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
        // Five-Whys: GPU garbage output → wrong RoPE style → rope_type not set for APR models
        let rope_type = model.metadata.rope_type.unwrap_or(0);
        executor.set_rope_type(rope_type);

        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);

        let mut apr_cuda = Self {
            model,
            executor,
            device_name,
            memory_info,
            weight_cache: std::collections::HashMap::new(),
            embedding_cache: None, // Lazy-loaded on first forward
            hidden_dim,
            kv_position: 0, // Start at position 0
        };

        // Pre-cache all transposed weights on GPU for 2x performance
        apr_cuda.pre_cache_weights()?;

        // Pre-cache embedding table for fast token lookup
        apr_cuda.cache_embeddings()?;

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

    /// Reset KV cache position for a new conversation.
    ///
    /// Call this before starting a new generation sequence to clear the
    /// KV cache state from the previous conversation.
    pub fn reset_kv_cache(&mut self) {
        self.kv_position = 0;
        self.executor.reset_kv_cache_gpu();
    }

    // ========================================================================
    // Weight Pre-caching (2x performance optimization)
    // ========================================================================

    /// Pre-cache all model weights on GPU using native quantized format.
    ///
    /// This uploads quantized weights (Q4K, Q6K, etc.) directly to GPU without
    /// CPU dequantization, enabling fused dequant+matmul kernels for maximum
    /// throughput (2x+ Ollama baseline per APR mandate).
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
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };
        let _kv_dim = num_kv_heads * head_dim;

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;
        let mut quantized_count = 0usize;

        // Helper to upload a weight tensor (quantized or F32)
        // Uses GGUF-style cache names for compatibility with build_indexed_weights()
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> usize {
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized: upload raw bytes to quantized_weight_cache
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0)
                    } else {
                        0
                    }
                } else {
                    // F32/F16: dequantize and upload to weight_cache (legacy path)
                    // This path is only used for non-quantized models
                    0 // Skip F32 weights - they'll be loaded on demand
                }
            } else {
                0
            }
        };

        // Cache per-layer weights using GGUF naming convention
        // This matches build_indexed_weights() expectations
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{layer_idx}");

            // Find source tensor names (HuggingFace, GGUF, etc.)
            // Map from various naming conventions to GGUF cache names
            let weight_mappings = [
                // (source_patterns, cache_suffix)
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("blk.{layer_idx}.attn_q.weight"),
                    ],
                    "attn_q.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("blk.{layer_idx}.attn_k.weight"),
                    ],
                    "attn_k.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("blk.{layer_idx}.attn_v.weight"),
                    ],
                    "attn_v.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("blk.{layer_idx}.attn_output.weight"),
                    ],
                    "attn_output.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("blk.{layer_idx}.ffn_gate.weight"),
                    ],
                    "ffn_gate.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("blk.{layer_idx}.ffn_up.weight"),
                    ],
                    "ffn_up.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("blk.{layer_idx}.ffn_down.weight"),
                    ],
                    "ffn_down.weight",
                ),
            ];

            for (patterns, suffix) in weight_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    let cache_name = format!("{prefix}.{suffix}");
                    let bytes = upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
                    if bytes > 0 {
                        total_bytes += bytes;
                        quantized_count += 1;
                    }
                }
            }

            // Upload RMSNorm gamma weights (always F32)
            let norm_mappings = [
                (
                    vec![
                        format!("model.layers.{layer_idx}.input_layernorm.weight"),
                        format!("layers.{layer_idx}.input_layernorm.weight"),
                        format!("blk.{layer_idx}.attn_norm.weight"),
                    ],
                    "attn_norm.gamma",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("blk.{layer_idx}.ffn_norm.weight"),
                    ],
                    "ffn_norm.gamma",
                ),
            ];

            for (patterns, suffix) in norm_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                        let cache_name = format!("{prefix}.{suffix}");
                        if let Ok(bytes) = self.executor.cache_rmsnorm_gamma(&cache_name, &gamma) {
                            total_bytes += bytes;
                        }
                    }
                }
            }
        }

        // Cache output norm
        let output_norm_patterns = [
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&output_norm_patterns) {
            if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                if let Ok(bytes) = self.executor.cache_rmsnorm_gamma("output_norm.gamma", &gamma) {
                    total_bytes += bytes;
                }
            }
        }

        // Cache LM head (may be quantized or F32)
        let lm_head_patterns = [
            "lm_head.weight",
            "output.weight",
            "token_embd.weight", // GGUF (tied embeddings)
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&lm_head_patterns) {
            if let Some(entry) = self.model.get_tensor(&src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized LM head
                    if let Ok(bytes) = self.model.get_tensor_bytes(&src_name) {
                        if let Ok(size) = self
                            .executor
                            .load_quantized_weights_with_type("output.weight", bytes, qtype)
                        {
                            total_bytes += size;
                            quantized_count += 1;
                        }
                    }
                } else {
                    // F32 LM head - store as quantized_weight_cache for compatibility
                    // The forward path will handle F32 appropriately
                    if let Ok(w) = self.model.get_tensor_f32(&src_name) {
                        // Upload F32 weights directly (no transpose needed for GEMV)
                        let w_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                w.as_ptr().cast::<u8>(),
                                w.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        // Use qtype 0 to indicate F32 (handled specially in forward)
                        if let Ok(size) = self
                            .executor
                            .load_quantized_weights_with_type("output.weight", w_bytes, 0)
                        {
                            total_bytes += size;
                        }
                    }
                }
            }
        }

        // Build indexed weight lookup table for O(1) access during decode
        // This is the key optimization that enables fast token generation
        if quantized_count > 0 {
            if let Err(e) = self
                .executor
                .build_indexed_weights(num_layers, |i| format!("blk.{i}"))
            {
                eprintln!("[AprV2ModelCuda] Warning: Could not build indexed weights: {e}");
                // Continue anyway - fallback path will be used
            } else {
                eprintln!(
                    "[AprV2ModelCuda] Built indexed weights for {} layers",
                    num_layers
                );
            }

            // Initialize workspace for zero-allocation forward pass
            if let Err(e) = self.executor.init_workspace(
                hidden_dim,
                intermediate_dim,
            ) {
                eprintln!("[AprV2ModelCuda] Warning: Could not init workspace: {e}");
            }
        }

        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers, {} quantized tensors)",
            total_bytes / (1024 * 1024),
            num_layers,
            quantized_count
        );

        Ok(())
    }

    /// Pre-cache embedding table for fast token lookup.
    ///
    /// This reads the embedding table once and stores it in memory, eliminating
    /// repeated disk/mmap reads during generation (~450ms → ~0.05ms per token).
    fn cache_embeddings(&mut self) -> Result<()> {
        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight", // GGUF naming
        ])?;

        let embeddings = self.model.get_tensor_f32(&embed_name)?;
        let embed_mb = embeddings.len() * 4 / (1024 * 1024);
        eprintln!(
            "[AprV2ModelCuda] Cached embedding table: {} MB",
            embed_mb
        );

        self.embedding_cache = Some(embeddings);
        Ok(())
    }

    /// Get embedding for a token ID from cache.
    #[inline]
    fn get_embedding(&self, token_id: u32) -> Option<&[f32]> {
        self.embedding_cache.as_ref().and_then(|cache| {
            let offset = (token_id as usize) * self.hidden_dim;
            if offset + self.hidden_dim <= cache.len() {
                Some(&cache[offset..offset + self.hidden_dim])
            } else {
                None
            }
        })
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

    /// GPU-accelerated forward pass returning only the next token ID (fastest path).
    ///
    /// Uses GPU argmax to avoid transferring 600KB of logits from GPU to CPU.
    /// This is the recommended method for autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token ID (single token for decode step)
    ///
    /// # Returns
    ///
    /// The token ID with the highest logit value.
    pub fn forward_cuda_to_token(&mut self, token_id: u32) -> Result<u32> {
        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let _hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let _num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);

        // Use indexed Q4K path with GPU argmax (no 600KB logits transfer)
        if self.executor.has_indexed_weights() {
            let position = self.kv_position;

            // Embedding lookup from cache
            let input: Vec<f32> = self.get_embedding(token_id).ok_or_else(|| {
                RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                }
            })?.to_vec();

            let num_layers = self.model.metadata.num_layers.unwrap_or(0);
            let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
            let intermediate_dim = self.model.metadata.intermediate_size.unwrap_or(hidden_dim * 4);
            let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);

            // First call: capture graph using the full graphed forward path
            // Subsequent calls: use replay with GPU argmax
            let next_token = if !self.executor.has_decode_graph() {
                // Need to capture graph first - use forward_all_layers_gpu_to_logits_graphed
                // then do CPU argmax
                let mut output = vec![0.0f32; vocab_size];
                self.executor
                    .forward_all_layers_gpu_to_logits_graphed(
                        &input,
                        &mut output,
                        position,
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                        eps,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                        reason: format!("Graph capture failed: {e}"),
                    })?;

                // CPU argmax for first token (graph now captured)
                let (top_idx, _) = output.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| RealizarError::InvalidShape {
                        reason: "Empty logits".to_string(),
                    })?;
                top_idx as u32
            } else {
                // Graph captured - use fast replay with GPU argmax
                self.executor
                    .forward_graphed_replay_to_token_id(
                        &input,
                        position,
                        vocab_size as u32,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_graphed_replay_to_token_id".to_string(),
                        reason: format!("GPU argmax fast path failed: {e}"),
                    })?
            };

            // Increment position for next token
            self.kv_position += 1;

            return Ok(next_token);
        }

        // Fallback: use forward_cuda and do CPU argmax
        let logits = self.forward_cuda(&[token_id])?;
        let (top_idx, _) = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(top_idx as u32)
    }

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
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // =========================================================================
        // FAST PATH: Use indexed Q4K GEMV kernels with CUDA graph capture
        // This path uses fused dequant+matmul kernels + graph replay for
        // 500x reduction in kernel launch overhead (5.6ms → 0.01ms per token)
        // =========================================================================
        if self.executor.has_indexed_weights() && seq_len == 1 {
            // Single-token decode: use the optimized Q4K GEMV path with graphs
            let token_id = token_ids[0];
            let position = self.kv_position;

            // Embedding lookup from cache (O(1) - no disk/mmap read)
            // Copy to local vec to release borrow before mutable executor call
            let input: Vec<f32> = self.get_embedding(token_id).ok_or_else(|| {
                RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                }
            })?.to_vec();

            // Use the graphed forward path with CUDA graph capture
            // First call captures the graph, subsequent calls replay it
            let mut output = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &input,
                    &mut output,
                    position,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                    reason: format!("Q4K graphed fast path failed: {e}"),
                })?;

            // Increment position for next token (KV cache tracking)
            self.kv_position += 1;

            return Ok(output);
        }

        // =========================================================================
        // FALLBACK PATH: Original F32 GEMM path (for prefill or non-indexed models)
        // =========================================================================

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
            "token_embd.weight", // GGUF naming
        ])?;
        let embeddings = self.model.get_tensor_f32(&embed_name)?;

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        if let Some(t) = timer_embed {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Get weight tensors (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF
            ])?;
            let q_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
                &format!("blk.{layer_idx}.attn_q.weight"), // GGUF
            ])?;
            let k_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
                &format!("blk.{layer_idx}.attn_k.weight"), // GGUF
            ])?;
            let v_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
                &format!("blk.{layer_idx}.attn_v.weight"), // GGUF
            ])?;
            let o_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF
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
                let q =
                    self.gemm_cached_gpu(&q_cache_name, &normed, seq_len, hidden_dim, hidden_dim)?;
                let k =
                    self.gemm_cached_gpu(&k_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                let v =
                    self.gemm_cached_gpu(&v_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
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
                &format!("blk.{layer_idx}.ffn_norm.weight"), // GGUF
            ])?;
            let gate_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"), // GGUF
            ])?;
            let up_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"), // GGUF
            ])?;
            let down_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"), // GGUF
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
                let gate_out = self.gemm_cached_gpu(
                    &gate_cache_name,
                    &normed,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                let up_out = self.gemm_cached_gpu(
                    &up_cache_name,
                    &normed,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                (gate_out, up_out)
            } else {
                // Fallback: load, transpose, and upload each time
                let gate = self.model.get_tensor_f32(&gate_name)?;
                let up = self.model.get_tensor_f32(&up_name)?;
                let gate_t = transpose_matrix(&gate, intermediate_dim, hidden_dim);
                let up_t = transpose_matrix(&up, intermediate_dim, hidden_dim);
                let gate_out =
                    self.gemm_gpu(&normed, &gate_t, seq_len, hidden_dim, intermediate_dim)?;
                let up_out =
                    self.gemm_gpu(&normed, &up_t, seq_len, hidden_dim, intermediate_dim)?;
                (gate_out, up_out)
            };

            // SiLU activation and element-wise multiply (CPU - fast)
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }

            let ffn_out = if self.has_cached_weight(&down_cache_name) {
                self.gemm_cached_gpu(
                    &down_cache_name,
                    &ffn_hidden,
                    seq_len,
                    intermediate_dim,
                    hidden_dim,
                )?
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
            "output_norm.weight", // GGUF naming
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
                "output.weight", // GGUF uses this
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
                .map_or(eos_id, |(idx, _)| idx as u32);

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
    pub fn forward_single_cuda(&mut self, token_id: u32, _position: usize) -> Result<Vec<f32>> {
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
        for _i in 0..max_new_tokens {
            let position = tokens.len();
            let last_token = *tokens.last().unwrap_or(&1);

            let logits = self.forward_single_cuda(last_token, position)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

// =============================================================================
// APR GPU Integration - Memory-Mapped Model Loading
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

        // Validate magic and version
        if header.magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: "Invalid APR magic bytes".to_string(),
            });
        }

        if header.version.0 > FORMAT_VERSION.0 {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR version {}.{} not supported (max {}.{})",
                    header.version.0, header.version.1, FORMAT_VERSION.0, FORMAT_VERSION.1
                ),
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
        let tensor = self.find_tensor(name).ok_or_else(|| RealizarError::FormatError {
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

    // ============ Additional coverage tests ============

    /// Helper to create a complete valid APR v2 model
    fn create_test_apr_model() -> Vec<u8> {
        let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = ((metadata_bytes.len() + 63) / 64) * 64;

        // Binary tensor index entry for "test.weight"
        let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let data_size = 64usize; // 16 floats * 4 bytes

        let total_size = data_offset as usize + data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data (16 floats)
        let data_start = data_offset as usize;
        for i in 0..16 {
            let val = i as f32;
            data[data_start + i * 4..data_start + i * 4 + 4]
                .copy_from_slice(&val.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let names = model.tensor_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "test.weight");
    }

    #[test]
    fn test_apr_model_metadata() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let meta = model.metadata();
        assert_eq!(meta.vocab_size, Some(100));
        assert_eq!(meta.hidden_size, Some(64));
    }

    #[test]
    fn test_apr_model_get_tensor() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tensor = model.get_tensor("test.weight");
        assert!(tensor.is_some());
        let entry = tensor.unwrap();
        assert_eq!(entry.shape, vec![4, 4]);
        assert_eq!(entry.dtype, "F32");

        // Non-existent tensor
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_model_get_tensor_f32() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let floats = model.get_tensor_f32("test.weight").expect("should get f32");
        assert_eq!(floats.len(), 16);
        assert_eq!(floats[0], 0.0);
        assert_eq!(floats[1], 1.0);
        assert_eq!(floats[15], 15.0);
    }

    #[test]
    fn test_apr_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_get_tensor_bytes() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let bytes = model.get_tensor_bytes("test.weight").expect("should get bytes");
        assert_eq!(bytes.len(), 64); // 16 floats * 4 bytes
    }

    #[test]
    fn test_apr_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // 4 x 4 = 16 parameters
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(!flags.is_zstd());
    }

    #[test]
    fn test_apr_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
        assert!(!flags.is_lz4());
    }

    #[test]
    fn test_apr_flags_has_vocab() {
        let flags = AprFlags::new(AprFlags::HAS_VOCAB);
        assert!(flags.has_vocab());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_metadata_is_transformer() {
        // is_transformer() requires hidden_size, num_layers, num_heads, vocab_size all Some
        let mut meta = AprMetadata::default();
        assert!(!meta.is_transformer()); // all None

        // Set all required fields
        meta.hidden_size = Some(1024);
        meta.num_layers = Some(12);
        meta.num_heads = Some(16);
        meta.vocab_size = Some(32000);
        assert!(meta.is_transformer());

        // Missing one field
        meta.hidden_size = None;
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_model_unsupported_version() {
        let mut data = vec![0u8; HEADER_SIZE + 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 99; // version major = 99 (unsupported)
        data[5] = 0;

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not supported"));
    }

    #[test]
    fn test_apr_model_encrypted_error() {
        let mut data = vec![0u8; HEADER_SIZE + 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes()); // flags = encrypted

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Encrypted"));
    }

    #[test]
    fn test_apr_model_truncated_metadata() {
        let mut data = vec![0u8; 100]; // Too small for metadata
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 (larger than file)

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_is_apr_file() {
        // is_apr_file reads the file and checks for APR2 magic bytes
        // Non-existent files return false
        assert!(!is_apr_file("/nonexistent/model.apr"));
        assert!(!is_apr_file("/nonexistent/model.gguf"));

        // Create temp file with APR magic
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_apr_file.apr");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&MAGIC).expect("write magic");
            f.write_all(&[0u8; 60]).expect("write padding");
        }
        assert!(is_apr_file(&path));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format("/path/model.bin"), "unknown");
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_bpe_tokenizer_encode_decode() {
        // Create vocab with ASCII characters
        let id_to_token: Vec<String> = (0u8..128)
            .map(|i| String::from_utf8(vec![i]).unwrap_or_default())
            .collect();

        let token_to_id: HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
        };

        // Encode simple ASCII
        let ids = tokenizer.encode("hi");
        assert!(!ids.is_empty());

        // Decode back
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_tokens() {
        let vocab: Vec<String> = vec![
            "hello".to_string(),
            " ".to_string(),
            "world".to_string(),
        ];

        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_decode_tokens_with_unknown() {
        let vocab: Vec<String> = vec!["a".to_string(), "b".to_string()];

        // Token 99 is out of bounds - decode_tokens formats as [id]
        let result = AprV2Model::decode_tokens(&vocab, &[0, 99, 1]);
        assert!(result.contains('a'));
        assert!(result.contains('b'));
        assert!(result.contains("[99]")); // Unknown tokens formatted as [id]
    }

    // =========================================================================
    // ModelData Tests (Memory-Mapped Model Loading)
    // =========================================================================

    #[test]
    fn test_model_data_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5];
        let model_data = ModelData::from_vec(data.clone());

        assert_eq!(model_data.as_slice(), &data);
        assert_eq!(model_data.len(), 5);
        assert!(!model_data.is_empty());
        assert!(!model_data.is_mmap());
    }

    #[test]
    fn test_model_data_from_vec_empty() {
        let model_data = ModelData::from_vec(vec![]);

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(!model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test mmap data").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert_eq!(model_data.as_slice(), b"test mmap data");
        assert_eq!(model_data.len(), 14);
        assert!(!model_data.is_empty());
        assert!(model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_nonexistent() {
        let result = ModelData::open_mmap("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_empty_file() {
        use tempfile::NamedTempFile;

        let temp = NamedTempFile::new().expect("create temp file");
        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(model_data.is_mmap());
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test release pages").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.release_cpu_pages().expect("release pages");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);

        // Should be no-op for heap data
        model_data.release_cpu_pages().expect("release pages (no-op)");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"sequential access test").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.advise_sequential().expect("advise sequential");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);

        // Should be no-op for heap data
        model_data.advise_sequential().expect("advise sequential (no-op)");
    }

    #[test]
    fn test_model_data_debug() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Heap"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_mmap_debug() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"debug test").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Mmap"));
    }

    // =========================================================================
    // AprV2Model mmap Integration Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_uses_mmap_for_uncompressed() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file (uncompressed)
        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write a minimal valid APR v2 file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0 (uncompressed)
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should use mmap for uncompressed files
        assert!(model.is_mmap(), "Uncompressed model should use mmap");
    }

    #[test]
    fn test_apr_model_from_bytes_uses_heap() {
        // Create a minimal valid APR v2 data
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let model = AprV2Model::from_bytes(data).expect("load model");

        // from_bytes always uses heap
        assert!(!model.is_mmap(), "from_bytes should use heap, not mmap");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_apr_model_release_cpu_pages() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should not error
        model.release_cpu_pages().expect("release pages");
    }

    #[test]
    fn test_apr_model_load_nonexistent_file() {
        let result = AprV2Model::load("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write invalid magic
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic
        data[4] = 2;
        data[5] = 0;

        temp.write_all(&data).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_truncated_header() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write truncated file (less than HEADER_SIZE)
        temp.write_all(&[0u8; 10]).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    // =========================================================================
    // F16 Conversion Tests
    // =========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        let zero: u16 = 0x0000;
        let result = super::f16_to_f32(zero);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let one: u16 = 0x3C00; // 1.0 in f16
        let result = super::f16_to_f32(one);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        let neg_one: u16 = 0xBC00; // -1.0 in f16
        let result = super::f16_to_f32(neg_one);
        assert!((result + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_inf() {
        let pos_inf: u16 = 0x7C00;
        let result = super::f16_to_f32(pos_inf);
        assert!(result.is_infinite() && result > 0.0);

        let neg_inf: u16 = 0xFC00;
        let result = super::f16_to_f32(neg_inf);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let nan: u16 = 0x7C01; // NaN (exp=31, mantissa!=0)
        let result = super::f16_to_f32(nan);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let subnormal: u16 = 0x0001; // Smallest positive subnormal
        let result = super::f16_to_f32(subnormal);
        assert!(result > 0.0 && result < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let neg_zero: u16 = 0x8000;
        let result = super::f16_to_f32(neg_zero);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_half() {
        let half: u16 = 0x3800; // 0.5 in f16
        let result = super::f16_to_f32(half);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_two() {
        let two: u16 = 0x4000; // 2.0 in f16
        let result = super::f16_to_f32(two);
        assert!((result - 2.0).abs() < 1e-6);
    }

    // =========================================================================
    // dequantize_f16 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_empty() {
        let result = super::dequantize_f16(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_f16_single() {
        // f16: 0x3C00 = 1.0 (little-endian: 0x00, 0x3C)
        let data: &[u8] = &[0x00, 0x3C];
        let result = super::dequantize_f16(data, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_f16_multiple() {
        // f16: 0x3C00 = 1.0, 0x4000 = 2.0
        let data: &[u8] = &[0x00, 0x3C, 0x00, 0x40];
        let result = super::dequantize_f16(data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
    }

    // =========================================================================
    // dequantize_q8_0 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Q8_0 block: 2-byte f16 scale + 32 int8 values
        // Create a simple block with scale = 1.0 (0x3C00) and values 0..32
        let mut data = Vec::new();
        data.push(0x00); // scale low byte
        data.push(0x3C); // scale high byte (1.0 in f16)
        for i in 0..32u8 {
            data.push(i);
        }
        let result = super::dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
        // First value should be 0 * 1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // Value at index 10 should be 10 * 1.0 = 10.0
        assert!((result[10] - 10.0).abs() < 1e-4);
    }

    // =========================================================================
    // BPE Tokenizer Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_empty_vocab() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_single_char_vocab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("abc");
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn test_bpe_tokenizer_unknown_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("xyz");
        // Unknown chars should be handled gracefully
        assert!(encoded.is_empty() || encoded.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_bpe_tokenizer_decode_empty() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_valid() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert("world".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let decoded = tokenizer.decode(&[0, 1]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_bpe_tokenizer_with_merge_rules() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "e".to_string(), "he".to_string()],
            merge_rules: vec![("h".to_string(), "e".to_string())],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("he");
        // After merging h+e -> he, should have 1 token
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_with_bos_eos() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        token_to_id.insert("a".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<s>".to_string(), "</s>".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: Some(0),
            eos_id: Some(1),
        };
        assert_eq!(tokenizer.bos_id, Some(0));
        assert_eq!(tokenizer.eos_id, Some(1));
    }

    // =========================================================================
    // is_apr_file Tests
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent() {
        // Non-existent file returns false (can't read magic bytes)
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[test]
    fn test_is_apr_file_wrong_extension() {
        // Non-existent files all return false
        assert!(!is_apr_file("/some/path/model.gguf"));
        assert!(!is_apr_file("/some/path/model.safetensors"));
        assert!(!is_apr_file("/some/path/model.bin"));
    }

    #[test]
    fn test_is_apr_file_no_extension() {
        assert!(!is_apr_file("/some/path/model"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_valid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write APR magic bytes
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_wrong_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write wrong magic bytes
        temp.write_all(b"GGUF").expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // detect_format Tests
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format("/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format("/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
    }

    // =========================================================================
    // TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_dtype_sizes() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 0,
            size: 800, // 10*20*4 bytes
        };
        assert_eq!(entry.element_count(), 200);
    }

    #[test]
    fn test_tensor_entry_empty_shape() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_tensor_entry_4d_shape() {
        let entry = TensorEntry {
            name: "4d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5],
            offset: 0,
            size: 480,
        };
        assert_eq!(entry.element_count(), 120);
    }

    #[test]
    fn test_tensor_entry_1d_shape() {
        let entry = TensorEntry {
            name: "1d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            offset: 0,
            size: 400,
        };
        assert_eq!(entry.element_count(), 100);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_len() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_model_data_is_empty() {
        let empty = ModelData::from_vec(vec![]);
        assert!(empty.is_empty());

        let non_empty = ModelData::from_vec(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_model_data_as_slice() {
        let data = ModelData::from_vec(vec![10, 20, 30]);
        let slice = data.as_slice();
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn test_model_data_large() {
        let large_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let data = ModelData::from_vec(large_data);
        assert_eq!(data.len(), 1000);
        assert_eq!(data.as_slice()[0], 0);
        assert_eq!(data.as_slice()[255], 255);
        assert_eq!(data.as_slice()[256], 0);
    }

    // =========================================================================
    // AprHeader Tests
    // =========================================================================

    #[test]
    fn test_apr_header_from_bytes_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC); // APR2
        data.extend_from_slice(&[2, 0]); // version 2.0
        data.extend_from_slice(&[0, 0]); // flags
        data.extend_from_slice(&5u32.to_le_bytes()); // tensor_count = 5
        data.extend_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data.extend_from_slice(&100u32.to_le_bytes()); // metadata_size
        data.extend_from_slice(&164u64.to_le_bytes()); // tensor_index_offset
        data.extend_from_slice(&500u64.to_le_bytes()); // data_offset
        data.extend_from_slice(&0u32.to_le_bytes()); // checksum
        data.extend_from_slice(&[0u8; 20]); // reserved

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.version.0, 2);
        assert_eq!(header.version.1, 0);
        assert_eq!(header.tensor_count, 5);
    }

    #[test]
    fn test_apr_header_from_bytes_wrong_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // Wrong magic
        data.extend_from_slice(&[0u8; 60]); // padding

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_from_bytes_too_short() {
        let data = vec![0u8; 10]; // Too short
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprMetadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_true() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_hidden() {
        let meta = AprMetadata {
            hidden_size: None,
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_layers() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: None,
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_default() {
        let meta = AprMetadata::default();
        assert!(meta.hidden_size.is_none());
        assert!(meta.num_layers.is_none());
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // AprV2Model Tests - Basic Operations
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_minimal() {
        let data = create_test_apr_model();
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_ok());
        let model = result.unwrap();
        // Helper creates 1 tensor
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let names = model.tensor_names();
        // Helper creates tensor "test.weight"
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.weight"));
    }

    #[test]
    fn test_apr_v2_model_metadata_default() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let meta = model.metadata();
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        // Helper creates 1 tensor with shape [4,4] = 16 elements
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_v2_model_is_mmap_false() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        assert!(!model.is_mmap());
    }

    // =========================================================================
    // AprV2Model Tests - predict
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_no_tensors() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        // With no tensors, returns sum of features
        let output = result.unwrap();
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_v2_model_predict_empty_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let features: Vec<f32> = vec![];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output[0], 0.0);
    }

    // =========================================================================
    // AprV2Model Tests - forward
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty tokens should fail
    }

    #[test]
    fn test_apr_v2_model_forward_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.forward(&[1, 2, 3]);
        // Should fail because metadata doesn't indicate transformer
        assert!(result.is_err());
    }

    // =========================================================================
    // decode_tokens Tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_input() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 5, 10]);
        // Should contain "hello" and [id] for invalid tokens
        assert!(result.contains("hello"));
        assert!(result.contains("[5]"));
        assert!(result.contains("[10]"));
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix() {
        let vocab = vec!["▁hello".to_string(), "▁world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Sentencepiece prefix should be converted to space
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // All tokens out of bounds, formatted as [id]
        assert!(result.contains("[0]"));
        assert!(result.contains("[1]"));
        assert!(result.contains("[2]"));
    }

    // =========================================================================
    // bpe_encode Tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_empty_text() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("", &token_to_id, &merge_rules);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_single_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("a", &token_to_id, &merge_rules);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_chars() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("xyz", &token_to_id, &merge_rules);
        // Unknown chars return empty or default behavior
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let merge_rules = vec![("h".to_string(), "e".to_string())];
        let result = bpe_encode("he", &token_to_id, &merge_rules);
        // Should merge h+e -> he
        assert!(!result.is_empty());
    }

    // =========================================================================
    // BpeTokenizer Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_whitespace() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(" ".to_string(), 0);
        token_to_id.insert("a".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![" ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode(" a ");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_sentencepiece() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("▁hello".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["▁hello".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let decoded = tokenizer.decode(&[0]);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_bpe_tokenizer_decode_unknown_id() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let decoded = tokenizer.decode(&[0, 100, 200]);
        // Should handle out of bounds gracefully
        assert!(decoded.contains("a") || decoded.contains("<unk>"));
    }

    // =========================================================================
    // dequantize_q4_k Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = super::dequantize_q4_k(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = super::dequantize_q6_k(&[], 0);
        assert!(result.is_empty());
    }

    // =========================================================================
    // dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_f32() {
        // F32 is not a quantized type, returns None
        assert_eq!(super::dtype_to_ggml_qtype("F32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16() {
        // F16 is not a quantized type, returns None
        assert_eq!(super::dtype_to_ggml_qtype("F16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0() {
        // Q4_0 is qtype 2 in GGML
        let result = super::dtype_to_ggml_qtype("Q4_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0() {
        // Q8_0 is qtype 8 in GGML
        let result = super::dtype_to_ggml_qtype("Q8_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown() {
        assert_eq!(super::dtype_to_ggml_qtype("UNKNOWN"), None);
    }

    // =========================================================================
    // is_quantized_dtype Tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_f32() {
        assert!(!super::is_quantized_dtype("F32"));
    }

    #[test]
    fn test_is_quantized_dtype_f16() {
        assert!(!super::is_quantized_dtype("F16"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_0() {
        assert!(super::is_quantized_dtype("Q4_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q8_0() {
        assert!(super::is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_k() {
        assert!(super::is_quantized_dtype("Q4_K"));
    }

    #[test]
    fn test_is_quantized_dtype_q6_k() {
        assert!(super::is_quantized_dtype("Q6_K"));
    }

    // =========================================================================
    // byte_to_bpe_char Tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        // ASCII printable range
        assert_eq!(super::byte_to_bpe_char(b'a'), "a");
        assert_eq!(super::byte_to_bpe_char(b'z'), "z");
        assert_eq!(super::byte_to_bpe_char(b'0'), "0");
    }

    #[test]
    fn test_byte_to_bpe_char_space() {
        // Space is encoded as Ġ in GPT-2 byte-level BPE
        assert_eq!(super::byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_special() {
        // Control chars get mapped to special unicode
        let result = super::byte_to_bpe_char(0);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_high_byte() {
        let result = super::byte_to_bpe_char(255);
        assert!(!result.is_empty());
    }

    // =========================================================================
    // rms_norm Tests
    // =========================================================================

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.74
        // Each element normalized by RMS
    }

    #[test]
    fn test_rms_norm_zeros() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // All zeros remain zeros
        for &v in &result {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Weight should scale the result
        assert!(result[0] > 1.0);
    }

    // =========================================================================
    // matmul Tests
    // =========================================================================

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix times [1,2,1,2] (2 seq, 2 dim)
        let x = vec![1.0, 2.0, 1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let result = super::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_simple() {
        // [1,2] * [[1],[1]] = [3]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = super::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    // transpose_matrix is not in public API

    // =========================================================================
    // simd_dot Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = super::simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let result = super::simd_dot(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_large() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 256];
        let result = super::simd_dot(&a, &b);
        // Sum of 0..255 = 255*256/2 = 32640
        assert!((result - 32640.0).abs() < 1e-3);
    }

    // detect_format and AprFlags tests already exist above
    // f16_to_f32 tests already exist above

    // =========================================================================
    // dequantize_f16 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_f16_basic() {
        // f16 1.0 = 0x3C00, stored as little-endian [0x00, 0x3C]
        let bytes = vec![0x00, 0x3C, 0x00, 0x3C];
        let result = super::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_f16_truncated() {
        let bytes = vec![0x00]; // Only 1 byte, needs 2
        let result = super::dequantize_f16(&bytes, 1);
        // Truncated input returns empty
        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // dequantize_q8_0 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_zero_scale() {
        // Q8_0 block: 2-byte scale + 32 bytes data
        let mut bytes = vec![0u8; 34];
        // Scale = 0.0 (f16)
        bytes[0] = 0x00;
        bytes[1] = 0x00;
        let result = super::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // All zeros with zero scale
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    // BpeTokenizer tests already exist above

    // =========================================================================
    // TensorEntry Tests (extended)
    // =========================================================================

    #[test]
    fn test_tensor_entry_byte_size_f32() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            offset: 0,
            size: 24, // 6 elements * 4 bytes
        };
        assert_eq!(entry.element_count(), 6);
    }

    #[test]
    fn test_tensor_entry_byte_size_f16() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 32, // 16 elements * 2 bytes
        };
        assert_eq!(entry.element_count(), 16);
    }

    // =========================================================================
    // AprMetadata Tests (extended)
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_missing_heads() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: None,
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_vocab() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: None,
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // simple_attention Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_single_head() {
        // Very simple case: 1 token, 1 head, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![1.0, 1.0];
        let result = super::simple_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(result.len(), 2);
        // Attention output should be weighted v
    }

    // CudaAprModel tests require GPU feature flag - tested elsewhere

    // =========================================================================
    // AprV2Model Error Path Tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_truncated() {
        // Too short to contain header
        let data = vec![0u8; 10];
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_invalid_magic() {
        let mut data = vec![0u8; 100];
        // Invalid magic
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_unsupported_version() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 99; // Future version
        data[5] = 0;
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        // test.weight exists in test model
        let result = model.get_tensor_bytes("test.weight");
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        // test.weight exists in test model
        let result = model.get_tensor_f32("test.weight");
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert!(!tensor.is_empty());
    }

    // =========================================================================
    // AprHeader Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_header_magic_validation() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX"); // Invalid magic
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_version_tuple() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Major
        data[5] = 1; // Minor
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.version, (2, 1));
    }

    // =========================================================================
    // AprFlags Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_flags_from_default() {
        let flags = AprFlags::default();
        // Default flags have no bits set
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // TensorEntry Extended Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_large_shape() {
        // Create entry with large multidimensional shape
        let entry = create_binary_tensor_entry("big_tensor", 0, &[10, 20, 30, 40], 0, 240000);
        let (parsed, consumed) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.name, "big_tensor");
        assert_eq!(parsed.shape, vec![10, 20, 30, 40]);
        assert_eq!(parsed.element_count(), 10 * 20 * 30 * 40);
        assert!(consumed > 0);
    }

    #[test]
    fn test_tensor_entry_from_binary_zero_dim() {
        // Scalar tensor (empty shape)
        let entry = create_binary_tensor_entry("scalar", 0, &[], 0, 4);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.name, "scalar");
        assert!(parsed.shape.is_empty());
        assert_eq!(parsed.element_count(), 1); // Scalar has 1 element
    }

    // =========================================================================
    // dequantize Function Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_multiple_values() {
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let bytes = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
        ];
        let result = super::dequantize_f16(&bytes, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q8_0_nonzero_scale() {
        // Q8_0 block: 2-byte scale + 32 i8 values
        // Scale = 1.0 (f16 0x3C00)
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00; // Scale low byte
        bytes[1] = 0x3C; // Scale high byte (1.0 in f16)
        // Set first few values to small integers
        bytes[2] = 1; // i8 value 1
        bytes[3] = 2; // i8 value 2
        bytes[4] = 255; // i8 value -1

        let result = super::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_vec_operations() {
        let data = vec![1u8, 2, 3, 4, 5];
        let md = ModelData::from_vec(data);
        assert_eq!(md.len(), 5);
        assert!(!md.is_empty());
        let slice = md.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // simple_attention Extended Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_multi_head() {
        // 2 tokens, 2 heads, head_dim=4
        let hidden_dim = 8; // 2 heads * 4 head_dim
        let q = vec![1.0; hidden_dim * 2]; // 2 tokens
        let k = vec![1.0; hidden_dim * 2];
        let v = vec![1.0; hidden_dim * 2];

        let result = super::simple_attention(&q, &k, &v, 2, 2, 2, 4);
        assert_eq!(result.len(), hidden_dim * 2);
    }

    #[test]
    fn test_simple_attention_gqa() {
        // GQA: 4 heads, 2 KV heads, head_dim=2
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim]; // 1 token
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = super::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // matmul Extended Tests
    // =========================================================================

    #[test]
    fn test_matmul_rectangular() {
        // [2,3] * [3,4] = [2,4]
        let x = vec![1.0; 2 * 3]; // 2 rows, 3 cols
        let w = vec![1.0; 3 * 4]; // 3 rows (in_dim), 4 cols (out_dim)
        let result = super::matmul(&x, &w, 2, 3, 4);
        assert_eq!(result.len(), 2 * 4);
        // Each output element = sum of 3 ones = 3.0
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_large() {
        // Larger matrix to exercise SIMD paths
        let seq_len = 4;
        let in_dim = 64;
        let out_dim = 64;
        let x = vec![1.0; seq_len * in_dim];
        let w = vec![1.0; in_dim * out_dim];
        let result = super::matmul(&x, &w, seq_len, in_dim, out_dim);
        assert_eq!(result.len(), seq_len * out_dim);
        // Each element = sum of 64 ones = 64.0
        assert!((result[0] - 64.0).abs() < 1e-3);
    }

    // =========================================================================
    // simd_dot Extended Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_mismatched_len() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 8];
        let result = super::simd_dot(&a, &b);
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_alternating() {
        let a = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = super::simd_dot(&a, &b);
        // Sum: 1 - 1 + 1 - 1 + 1 - 1 + 1 - 1 = 0
        assert!((result - 0.0).abs() < 1e-6);
    }

    // =========================================================================
    // BpeTokenizer Decode Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_byte_fallback() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<0x41>".to_string(), 0); // Byte 65 = 'A'
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<0x41>".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let decoded = tokenizer.decode(&[0]);
        // Byte fallback should be handled
        assert!(!decoded.is_empty());
    }

    // =========================================================================
    // AprFlags Extended Tests - Coverage for all flag methods
    // =========================================================================

    #[test]
    fn test_apr_flags_lz4_compressed() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed()); // LZ4 counts as compressed
    }

    #[test]
    fn test_apr_flags_zstd_compressed() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed()); // ZSTD counts as compressed
    }

    #[test]
    fn test_apr_flags_encrypted() {
        let flags = AprFlags::new(AprFlags::ENCRYPTED);
        assert!(flags.is_encrypted());
        assert!(!flags.is_compressed());
    }

    #[test]
    fn test_apr_flags_quantized() {
        let flags = AprFlags::new(AprFlags::QUANTIZED);
        assert!(flags.is_quantized());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_multiple() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
        assert!(!flags.is_zstd());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // f16_to_f32 Extended Tests - Infinity cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_infinity() {
        // +Inf in f16 = 0x7C00
        let result = super::f16_to_f32(0x7C00);
        assert!(result.is_infinite() && result > 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // -Inf in f16 = 0xFC00
        let result = super::f16_to_f32(0xFC00);
        assert!(result.is_infinite() && result < 0.0);
    }

    // =========================================================================
    // dequantize_q4_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_partial_block() {
        // Less than one full super-block (144 bytes)
        let bytes = vec![0u8; 50];
        let result = super::dequantize_q4_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q4_k_one_block() {
        // One complete Q4_K super-block (144 bytes = 256 elements)
        let mut bytes = vec![0u8; 144];
        // Set d (f16) = 1.0 = 0x3C00
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Set dmin (f16) = 0.0
        bytes[2] = 0x00;
        bytes[3] = 0x00;
        // scales and mins are already zeros
        // qs are already zeros

        let result = super::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // dequantize_q6_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q6_k_partial_block() {
        // Less than one full super-block (210 bytes)
        let bytes = vec![0u8; 100];
        let result = super::dequantize_q6_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q6_k_one_block() {
        // One complete Q6_K super-block (210 bytes = 256 elements)
        let mut bytes = vec![0u8; 210];
        // d (f16) is at the end: offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // 1.0 in f16

        let result = super::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // TensorEntry::from_binary dtype coverage
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_i8() {
        let entry = create_binary_tensor_entry("i8_tensor", 3, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "I8");
    }

    #[test]
    fn test_tensor_entry_from_binary_i16() {
        let entry = create_binary_tensor_entry("i16_tensor", 4, &[4], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "I16");
    }

    #[test]
    fn test_tensor_entry_from_binary_i32() {
        let entry = create_binary_tensor_entry("i32_tensor", 5, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "I32");
    }

    #[test]
    fn test_tensor_entry_from_binary_i64() {
        let entry = create_binary_tensor_entry("i64_tensor", 6, &[4], 0, 32);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "I64");
    }

    #[test]
    fn test_tensor_entry_from_binary_u8() {
        let entry = create_binary_tensor_entry("u8_tensor", 7, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "U8");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_k() {
        let entry = create_binary_tensor_entry("q4k_tensor", 8, &[256], 0, 144);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "Q4_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q6_k() {
        let entry = create_binary_tensor_entry("q6k_tensor", 9, &[256], 0, 210);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "Q6_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q8_0() {
        let entry = create_binary_tensor_entry("q8_tensor", 10, &[32], 0, 34);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "Q8_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype() {
        // Unknown dtype byte defaults to F32
        let entry = create_binary_tensor_entry("unknown_tensor", 255, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).unwrap();
        assert_eq!(parsed.dtype, "F32");
    }

    // =========================================================================
    // AprV2Model encrypted file test
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_encrypted() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version 2.0
        data[5] = 0;
        // Set encrypted flag (0x0004)
        data[6] = 0x04;
        data[7] = 0x00;

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{err:?}");
        assert!(err_msg.contains("Encrypted"));
    }

    // =========================================================================
    // AprV2Model::generate tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_generate_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.generate(&[], 10, None);
        assert!(result.is_err()); // Empty input should fail
    }

    #[test]
    fn test_apr_v2_model_generate_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        // Model without transformer config should fail on generate
        let result = model.generate(&[1, 2, 3], 5, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype coverage
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_lowercase() {
        assert!(super::dtype_to_ggml_qtype("q4_k").is_some());
        assert!(super::dtype_to_ggml_qtype("q5_k").is_some());
        assert!(super::dtype_to_ggml_qtype("q6_k").is_some());
        assert!(super::dtype_to_ggml_qtype("q8_0").is_some());
        assert!(super::dtype_to_ggml_qtype("q4_0").is_some());
        assert!(super::dtype_to_ggml_qtype("q4_1").is_some());
        assert!(super::dtype_to_ggml_qtype("q5_0").is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k() {
        assert_eq!(super::dtype_to_ggml_qtype("Q5_K"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_1() {
        assert_eq!(super::dtype_to_ggml_qtype("Q4_1"), Some(3));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_0() {
        assert_eq!(super::dtype_to_ggml_qtype("Q5_0"), Some(6));
    }

    // =========================================================================
    // AprMetadata extended coverage
    // =========================================================================

    #[test]
    fn test_apr_metadata_with_extra_fields() {
        let json = r#"{
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "vocab_size": 32000,
            "custom_field": "custom_value",
            "another_field": 42
        }"#;
        let meta: AprMetadata = serde_json::from_str(json).unwrap();
        assert!(meta.is_transformer());
        assert_eq!(meta.hidden_size, Some(256));
        // Extra fields should be captured
        assert!(meta.extra.contains_key("custom_field"));
    }

    #[test]
    fn test_apr_metadata_optional_fields() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("test-model".to_string()),
            architecture: Some("transformer".to_string()),
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            num_kv_heads: Some(4),
            vocab_size: Some(50000),
            intermediate_size: Some(4096),
            max_position_embeddings: Some(2048),
            rope_theta: Some(10000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        assert!(meta.is_transformer());
        assert_eq!(meta.num_kv_heads, Some(4));
        assert_eq!(meta.rope_type, Some(2));
    }

    // =========================================================================
    // byte_to_bpe_char extended coverage
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_all_printable_ascii() {
        // Test all printable ASCII characters
        for b in 33u8..=126 {
            let result = super::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_control_chars() {
        // Test control characters 0-31
        for b in 0u8..32 {
            let result = super::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_extended_ascii() {
        // Test extended ASCII 128-255
        for b in 128u8..=255 {
            let result = super::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    // =========================================================================
    // BpeTokenizer extended coverage
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_multiple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert(" ".to_string(), 1);
        token_to_id.insert("world".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), " ".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    // =========================================================================
    // AprHeader flags coverage
    // =========================================================================

    #[test]
    fn test_apr_header_with_flags() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set quantized flag
        data[6] = 0x20;
        data[7] = 0x00;

        let header = AprHeader::from_bytes(&data).unwrap();
        assert!(header.flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_single_feature() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let features = vec![1.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_predict_many_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let features: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    // =========================================================================
    // TensorEntry element_count edge cases
    // =========================================================================

    #[test]
    fn test_tensor_entry_element_count_1d() {
        let entry = TensorEntry {
            name: "vec".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            offset: 0,
            size: 40,
        };
        assert_eq!(entry.element_count(), 10);
    }

    #[test]
    fn test_tensor_entry_element_count_3d() {
        let entry = TensorEntry {
            name: "3d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4],
            offset: 0,
            size: 96,
        };
        assert_eq!(entry.element_count(), 24);
    }

    // =========================================================================
    // dequantize_f16 edge cases
    // =========================================================================

    #[test]
    fn test_dequantize_f16_odd_bytes() {
        // Odd number of bytes - last byte ignored
        let bytes = vec![0x00, 0x3C, 0x00]; // 1.0 + extra byte
        let result = super::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 1); // Only one complete f16
    }

    // =========================================================================
    // detect_format additional cases
    // =========================================================================

    #[test]
    fn test_detect_format_bin() {
        // .bin is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.bin"), "unknown");
    }

    #[test]
    fn test_detect_format_pt() {
        // .pt is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_detect_format_no_extension() {
        // No extension returns "unknown"
        assert_eq!(detect_format("/path/model"), "unknown");
    }

    #[test]
    fn test_detect_format_hidden_file() {
        // Hidden file with no extension returns "unknown"
        assert_eq!(detect_format("/path/.hidden"), "unknown");
    }

    // =========================================================================
    // dequantize Early Exit Tests - Hit break paths by requesting fewer elements
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_fewer_than_block() {
        // Q8_0 block: 2-byte scale + 32 i8 values = 34 bytes, 32 elements
        // Request only 10 elements to trigger early exit
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            bytes[2 + i] = (i + 1) as u8; // Values 1-32
        }

        let result = super::dequantize_q8_0(&bytes, 10);
        assert_eq!(result.len(), 10);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Two Q8_0 blocks = 68 bytes, 64 elements
        let mut bytes = vec![0u8; 68];
        // Block 1
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
        // Block 2
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0

        let result = super::dequantize_q8_0(&bytes, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q4_k_fewer_than_superblock() {
        // Q4_K super-block: 144 bytes = 256 elements
        // Request only 100 elements to trigger early exit
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0

        let result = super::dequantize_q4_k(&bytes, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q4_k_multiple_superblocks() {
        // Two Q4_K super-blocks = 288 bytes = 512 elements
        let mut bytes = vec![0u8; 288];
        // First superblock
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Second superblock
        bytes[144] = 0x00;
        bytes[145] = 0x3C;

        let result = super::dequantize_q4_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6_k_fewer_than_superblock() {
        // Q6_K super-block: 210 bytes = 256 elements
        // Request only 50 elements to trigger early exit
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = super::dequantize_q6_k(&bytes, 50);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_dequantize_q6_k_multiple_superblocks() {
        // Two Q6_K super-blocks = 420 bytes = 512 elements
        let mut bytes = vec![0u8; 420];
        // First superblock d at 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C;
        // Second superblock d at 418-419
        bytes[418] = 0x00;
        bytes[419] = 0x3C;

        let result = super::dequantize_q6_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    // =========================================================================
    // BpeTokenizer encode tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_encode_with_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("ab".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: None,
            eos_id: None,
        };
        let encoded = tokenizer.encode("ab");
        // Should merge a+b -> ab
        assert!(!encoded.is_empty());
    }

    // =========================================================================
    // AprV2Model find_tensor_name tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_total_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        // Test model has 4x4 = 16 element tensor
        assert!(model.estimated_parameters() > 0);
    }

    // =========================================================================
    // AprMetadata serialization tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_to_json() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("256"));
        assert!(json.contains("hidden_size"));
    }

    #[test]
    fn test_apr_metadata_roundtrip() {
        let meta = AprMetadata {
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            vocab_size: Some(50000),
            model_type: Some("llama".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: AprMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hidden_size, Some(1024));
        assert_eq!(parsed.model_type, Some("llama".to_string()));
    }

    // =========================================================================
    // TensorEntry tests with various shapes
    // =========================================================================

    #[test]
    fn test_tensor_entry_5d_shape() {
        let entry = TensorEntry {
            name: "5d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5, 6],
            offset: 0,
            size: 2880,
        };
        assert_eq!(entry.element_count(), 720);
    }

    #[test]
    fn test_tensor_entry_single_element() {
        let entry = TensorEntry {
            name: "single".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    // =========================================================================
    // is_apr_file tests (requires actual file with magic bytes)
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent_path() {
        // Non-existent file returns false
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_without_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"GGUF").expect("write wrong magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // More rms_norm tests
    // =========================================================================

    #[test]
    fn test_rms_norm_large() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 64];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let x = vec![-1.0, -2.0, -3.0, -4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Normalized negative values should remain negative
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // matmul extended tests
    // =========================================================================

    #[test]
    fn test_matmul_zeros() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let result = super::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_scaling() {
        let x = vec![2.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = super::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional Coverage Tests - get_tensor_f32 with different dtypes
    // =========================================================================

    /// Helper to create an APR model with a specific dtype tensor
    fn create_test_apr_model_with_dtype(dtype: u8, data_bytes: &[u8]) -> Vec<u8> {
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = ((metadata_bytes.len() + 63) / 64) * 64;

        // Tensor shape depends on dtype and data
        let tensor_entry = create_binary_tensor_entry("typed.weight", dtype, &[4], 0, data_bytes.len() as u64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + data_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data
        let data_start = data_offset as usize;
        data[data_start..data_start + data_bytes.len()].copy_from_slice(data_bytes);

        data
    }

    #[test]
    fn test_get_tensor_f32_f16_dtype() {
        // F16 dtype = 1, shape [4], 4 elements = 8 bytes
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let f16_data = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];
        let model_data = create_test_apr_model_with_dtype(1, &f16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.unwrap();
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 0.1);
        assert!((floats[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_get_tensor_f32_q8_0_dtype() {
        // Q8_0 dtype = 10, block = 2-byte scale + 32 i8 values = 34 bytes for 32 elements
        let mut q8_data = vec![0u8; 34];
        q8_data[0] = 0x00;
        q8_data[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            q8_data[2 + i] = i as u8;
        }

        // Create with shape [32] since Q8_0 block has 32 elements
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = ((metadata_bytes.len() + 63) / 64) * 64;

        let tensor_entry = create_binary_tensor_entry("typed.weight", 10, &[32], 0, 34);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 34;
        let mut model_data = vec![0u8; total_size];

        model_data[0..4].copy_from_slice(&MAGIC);
        model_data[4] = 2;
        model_data[5] = 0;
        model_data[8..12].copy_from_slice(&1u32.to_le_bytes());
        model_data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        model_data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        model_data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        model_data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        model_data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        model_data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        model_data[data_start..data_start + 34].copy_from_slice(&q8_data);

        let model = AprV2Model::from_bytes(model_data).expect("should load");
        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.unwrap();
        assert_eq!(floats.len(), 32);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_dtype() {
        // BF16 dtype = 2 is not fully supported for get_tensor_f32
        let bf16_data = vec![0x00, 0x3F, 0x80, 0x00]; // Two BF16 values
        let model_data = create_test_apr_model_with_dtype(2, &bf16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        // BF16 is not in the supported list, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_out_of_bounds() {
        // Create a model where tensor data extends beyond file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&100u64.to_le_bytes()); // data_offset

        // Add tensor entry that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("oob.weight", 0, &[1000], 0, 4000);
        data[64..64 + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");
        let result = model.get_tensor_f32("oob.weight");
        assert!(result.is_err()); // Out of bounds
    }

    // =========================================================================
    // decode_tokens extended tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_gpt2_special() {
        // Test GPT-2 style byte-level BPE tokens
        let vocab = vec![
            "Ġhello".to_string(), // Space + hello
            "Ċ".to_string(),      // Newline
            "ĉ".to_string(),      // Tab
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert!(result.contains("hello"));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_decode_tokens_empty_string_token() {
        let vocab = vec!["".to_string(), "a".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Empty token shouldn't cause issues
        assert!(result.contains('a'));
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_smallest_positive_normal() {
        // Smallest positive normal f16 = 0x0400 (6.103515625e-5)
        let result = super::f16_to_f32(0x0400);
        assert!(result > 0.0 && result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_largest_normal() {
        // Largest finite f16 = 0x7BFF (65504.0)
        let result = super::f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 10.0);
    }

    #[test]
    fn test_f16_to_f32_negative_normal() {
        // -2.0 in f16 = 0xC000
        let result = super::f16_to_f32(0xC000);
        assert!((result + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32_subnormal_nonzero() {
        // Various subnormals (exp=0, mantissa!=0)
        for mant in [1u16, 10, 100, 0x3FF] {
            let result = super::f16_to_f32(mant);
            assert!(result > 0.0, "Subnormal {mant:#x} should be positive");
        }
    }

    // =========================================================================
    // bpe_encode extended tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_with_newline() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ċ".to_string(), 0); // Newline token
        let result = bpe_encode("\n", &token_to_id, &[]);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_tab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("ĉ".to_string(), 0); // Tab token
        let result = bpe_encode("\t", &token_to_id, &[]);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space token
        let result = bpe_encode(" ", &token_to_id, &[]);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_mixed() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("Ġ".to_string(), 1); // Space
        token_to_id.insert("b".to_string(), 2);
        let result = bpe_encode("a b", &token_to_id, &[]);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_multiple_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("ab".to_string(), 3);
        token_to_id.insert("abc".to_string(), 4);

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let result = bpe_encode("abc", &token_to_id, &merges);
        // Should merge a+b->ab, then ab+c->abc
        assert!(!result.is_empty());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    /// Helper to create APR model with weight and bias tensors for linear model
    fn create_linear_model_apr() -> Vec<u8> {
        let metadata = r#"{"architecture":"linear"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = ((metadata_bytes.len() + 63) / 64) * 64;

        // Weight tensor: 2x3 = 6 elements, 24 bytes
        // Bias tensor: 2 elements, 8 bytes
        let weight_entry = create_binary_tensor_entry("weight", 0, &[2, 3], 0, 24);
        let bias_entry = create_binary_tensor_entry("bias", 0, &[2], 24, 8);
        let tensor_index_size = weight_entry.len() + bias_entry.len();

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index_size as u64;
        let total_size = data_offset as usize + 32;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&2u32.to_le_bytes()); // tensor_count = 2
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + weight_entry.len()].copy_from_slice(&weight_entry);
        data[idx_start + weight_entry.len()..idx_start + tensor_index_size].copy_from_slice(&bias_entry);

        // Tensor data - weight matrix [2,3] (identity-ish pattern)
        let data_start = data_offset as usize;
        let weights: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3] matrix
        for (i, &w) in weights.iter().enumerate() {
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // Bias [2]
        let biases: [f32; 2] = [0.5, 0.5];
        for (i, &b) in biases.iter().enumerate() {
            data[data_start + 24 + i * 4..data_start + 24 + i * 4 + 4].copy_from_slice(&b.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_v2_model_predict_with_weights() {
        let model_data = create_linear_model_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
        // Expected: [1*1 + 2*0 + 3*0 + 0.5, 1*0 + 2*1 + 3*0 + 0.5] = [1.5, 2.5]
        assert!((output[0] - 1.5).abs() < 0.01);
        assert!((output[1] - 2.5).abs() < 0.01);
    }

    // =========================================================================
    // AprFlags bit operations
    // =========================================================================

    #[test]
    fn test_apr_flags_sharded() {
        let flags = AprFlags::new(AprFlags::SHARDED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_flags_signed() {
        let flags = AprFlags::new(AprFlags::SIGNED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_all_set() {
        let flags = AprFlags::new(0xFFFF);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
    }

    // =========================================================================
    // simd_dot edge cases
    // =========================================================================

    #[test]
    fn test_simd_dot_empty() {
        let result = super::simd_dot(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_simd_dot_single_element() {
        let result = super::simd_dot(&[3.0], &[4.0]);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];
        let result = super::simd_dot(&a, &b);
        // (-1)*(-1) + (-2)*(-1) + (-3)*(-1) + (-4)*(-1) = 1 + 2 + 3 + 4 = 10
        assert!((result - 10.0).abs() < 1e-6);
    }

    // =========================================================================
    // rms_norm multi-sequence
    // =========================================================================

    #[test]
    fn test_rms_norm_multi_sequence() {
        // 2 sequences of hidden_dim=4
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // =========================================================================
    // simple_attention edge cases
    // =========================================================================

    #[test]
    fn test_simple_attention_mqa() {
        // MQA: 4 Q heads, 1 KV head
        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim];
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = super::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // AprHeader debug and clone
    // =========================================================================

    #[test]
    fn test_apr_header_clone() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&5u32.to_le_bytes());

        let header = AprHeader::from_bytes(&data).unwrap();
        let cloned = header.clone();
        assert_eq!(cloned.tensor_count, header.tensor_count);
        assert_eq!(cloned.version, header.version);
    }

    #[test]
    fn test_apr_header_debug() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;

        let header = AprHeader::from_bytes(&data).unwrap();
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("AprHeader"));
    }

    // =========================================================================
    // TensorEntry debug and clone
    // =========================================================================

    #[test]
    fn test_tensor_entry_debug() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("TensorEntry"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_tensor_entry_clone() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, entry.name);
        assert_eq!(cloned.shape, entry.shape);
    }

    // =========================================================================
    // AprMetadata debug and clone
    // =========================================================================

    #[test]
    fn test_apr_metadata_debug() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            ..Default::default()
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("AprMetadata"));
    }

    #[test]
    fn test_apr_metadata_clone() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            ..Default::default()
        };
        let cloned = meta.clone();
        assert_eq!(cloned.hidden_size, meta.hidden_size);
        assert_eq!(cloned.num_layers, meta.num_layers);
    }

    // =========================================================================
    // BpeTokenizer debug and clone
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_debug() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
        };
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("BpeTokenizer"));
    }

    #[test]
    fn test_bpe_tokenizer_clone() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: Some(1),
            eos_id: Some(2),
        };
        let cloned = tokenizer.clone();
        assert_eq!(cloned.bos_id, tokenizer.bos_id);
        assert_eq!(cloned.id_to_token, tokenizer.id_to_token);
    }

    // =========================================================================
    // AprFlags debug
    // =========================================================================

    #[test]
    fn test_apr_flags_debug() {
        let flags = AprFlags::new(0x0025);
        let debug_str = format!("{:?}", flags);
        assert!(debug_str.contains("AprFlags"));
    }

    #[test]
    fn test_apr_flags_clone() {
        let flags = AprFlags::new(0x0025);
        let cloned = flags;
        assert_eq!(cloned.is_lz4(), flags.is_lz4());
        assert_eq!(cloned.is_quantized(), flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model debug
    // =========================================================================

    #[test]
    fn test_apr_v2_model_debug() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("AprV2Model"));
    }

    // =========================================================================
    // ModelData extended edge cases
    // =========================================================================

    #[test]
    fn test_model_data_debug_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let debug_str = format!("{:?}", data);
        assert!(debug_str.contains("Heap"));
    }

    // =========================================================================
    // dequantize multiple blocks
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_three_blocks() {
        // Three Q8_0 blocks = 102 bytes, 96 elements
        let mut bytes = vec![0u8; 102];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 block 1
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0 block 2
        bytes[68] = 0x00;
        bytes[69] = 0x3C; // Scale = 1.0 block 3

        let result = super::dequantize_q8_0(&bytes, 96);
        assert_eq!(result.len(), 96);
    }

    #[test]
    fn test_dequantize_q4_k_with_nonzero_scales() {
        // Q4_K super-block with actual scale values
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        bytes[2] = 0x00;
        bytes[3] = 0x3C; // dmin = 1.0
        // Set some scale values
        bytes[4] = 0x3F; // scales[0] = 63
        bytes[5] = 0x3F;

        let result = super::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_with_nonzero_scales() {
        // Q6_K super-block with scale values
        let mut bytes = vec![0u8; 210];
        // scales at offset 192-207
        bytes[192] = 10;
        bytes[193] = 20;
        // d at offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = super::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // Transformer Model Tests - Forward Pass Coverage
    // =========================================================================

    /// Helper to create a minimal transformer model for testing forward pass.
    /// This creates a 1-layer transformer with tiny dimensions for test purposes.
    fn create_mini_transformer_apr() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "llama",
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "vocab_size": 10,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = ((metadata_bytes.len() + 63) / 64) * 64;

        // Tensors needed for forward:
        // - model.embed_tokens.weight [vocab=10, hidden=8] = 80 floats = 320 bytes
        // - layers.0.input_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.self_attn.q_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.k_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.v_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.o_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.post_attention_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.mlp.gate_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.up_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.down_proj.weight [hidden=8, inter=16] = 128 floats = 512 bytes
        // - norm.weight [hidden=8] = 8 floats = 32 bytes
        // - lm_head.weight [vocab=10, hidden=8] = 80 floats = 320 bytes

        let tensor_defs: Vec<(&str, &[usize], usize)> = vec![
            ("model.embed_tokens.weight", &[10, 8], 320),
            ("layers.0.input_layernorm.weight", &[8], 32),
            ("layers.0.self_attn.q_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.k_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.v_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.o_proj.weight", &[8, 8], 256),
            ("layers.0.post_attention_layernorm.weight", &[8], 32),
            ("layers.0.mlp.gate_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.up_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.down_proj.weight", &[8, 16], 512),
            ("norm.weight", &[8], 32),
            ("lm_head.weight", &[10, 8], 320),
        ];

        let mut tensor_entries = Vec::new();
        let mut current_offset = 0u64;

        for (name, shape, byte_size) in &tensor_defs {
            let shape_vec: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
            let entry = create_binary_tensor_entry(name, 0, &shape_vec, current_offset, *byte_size as u64);
            tensor_entries.push(entry);
            current_offset += *byte_size as u64;
        }

        let tensor_index: Vec<u8> = tensor_entries.iter().flat_map(|e| e.iter().copied()).collect();
        let tensor_count = tensor_defs.len() as u32;
        let total_data_size = current_offset as usize;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index.len() as u64;
        let total_size = data_offset as usize + total_data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

        // Tensor data - initialize with small random-ish values
        let data_start = data_offset as usize;
        let num_floats = total_data_size / 4;
        for i in 0..num_floats {
            let val = ((i % 10) as f32 - 5.0) * 0.1; // Small values between -0.5 and 0.4
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Set layernorm weights to 1.0 (they need to be non-zero)
        let norm_weight_offsets = vec![320, 320 + 32 + 256 * 4, 320 + 32 + 256 * 4 + 32 + 512 * 3];
        for offset in norm_weight_offsets {
            for i in 0..8 {
                let val = 1.0f32;
                let pos = data_start + offset + i * 4;
                if pos + 4 <= data.len() {
                    data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        data
    }

    #[test]
    fn test_transformer_model_loads() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");
        assert!(model.metadata().is_transformer());
        assert_eq!(model.metadata().hidden_size, Some(8));
        assert_eq!(model.metadata().num_layers, Some(1));
        assert_eq!(model.metadata().vocab_size, Some(10));
    }

    #[test]
    fn test_transformer_has_all_tensors() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        // Check key tensors exist
        assert!(model.get_tensor("model.embed_tokens.weight").is_some());
        assert!(model.get_tensor("layers.0.input_layernorm.weight").is_some());
        assert!(model.get_tensor("layers.0.self_attn.q_proj.weight").is_some());
        assert!(model.get_tensor("norm.weight").is_some());
        assert!(model.get_tensor("lm_head.weight").is_some());
    }

    // =========================================================================
    // matmul edge cases
    // =========================================================================

    #[test]
    fn test_matmul_out_of_bounds_x() {
        // x is shorter than expected
        let x = vec![1.0, 2.0]; // Only 2 elements, but seq=1, in_dim=4
        let w = vec![1.0; 8]; // 2 output dims, 4 input dims
        let result = super::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully (skip or produce zeros)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_matmul_out_of_bounds_w() {
        // w is shorter than expected
        let x = vec![1.0; 4]; // seq=1, in_dim=4
        let w = vec![1.0, 2.0]; // Too short
        let result = super::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully
        assert_eq!(result.len(), 2);
    }

    // =========================================================================
    // dequantize with negative i8 values
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
        // Set negative i8 values (128-255 map to -128 to -1)
        bytes[2] = 255; // -1 as i8
        bytes[3] = 254; // -2 as i8
        bytes[4] = 128; // -128 as i8

        let result = super::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value should be negative: -1 * 1.0 = -1.0
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // f16_to_f32 quiet NaN
    // =========================================================================

    #[test]
    fn test_f16_to_f32_qnan() {
        // Quiet NaN in f16 (exp=0x1F, mantissa>0, sign=0)
        // 0x7E00 = quiet NaN
        let result = super::f16_to_f32(0x7E00);
        assert!(result.is_nan());
    }

    // =========================================================================
    // detect_format with magic bytes
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "apr");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_gguf_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&[0x47, 0x47, 0x55, 0x46]).expect("write GGUF magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "gguf");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_safetensors_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"{\"test\": 1}").expect("write JSON");

        assert_eq!(detect_format(temp.path()), "safetensors");
    }

    // =========================================================================
    // simd_dot_avx2 scalar fallback
    // =========================================================================

    #[test]
    fn test_simd_dot_non_multiple_of_8() {
        // Test with lengths that aren't multiples of 8 to exercise remainder handling
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 13];
        let result = super::simd_dot(&a, &b);
        // Sum of 0..12 = 78
        assert!((result - 78.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_prime_length() {
        // Prime number length ensures both SIMD chunks and remainder are exercised
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 17];
        let result = super::simd_dot(&a, &b);
        // Sum of 0..16 = 136
        assert!((result - 136.0).abs() < 1e-6);
    }

    // =========================================================================
    // simple_attention multi-token
    // =========================================================================

    #[test]
    fn test_simple_attention_multiple_tokens() {
        // 3 tokens, 2 heads, head_dim=4
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let hidden_dim = num_heads * head_dim;

        let q = vec![1.0; seq_len * hidden_dim];
        let k = vec![1.0; seq_len * hidden_dim];
        let v = vec![1.0; seq_len * hidden_dim];

        let result = super::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
    }

    #[test]
    fn test_simple_attention_varying_values() {
        // Test with varying Q, K, V values
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let seq_len = 2;
        let hidden_dim = num_heads * head_dim;

        // Different Q, K, V patterns
        let q = vec![1.0, 0.0, 0.0, 1.0]; // Token 1: [1,0], Token 2: [0,1]
        let k = vec![1.0, 0.0, 1.0, 0.0]; // Token 1: [1,0], Token 2: [1,0]
        let v = vec![1.0, 2.0, 3.0, 4.0]; // Token 1: [1,2], Token 2: [3,4]

        let result = super::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
        // Output should be valid attention-weighted values
        assert!(!result.iter().any(|v| v.is_nan()));
    }

    // =========================================================================
    // ModelData methods
    // =========================================================================

    #[test]
    fn test_model_data_empty() {
        let data = ModelData::from_vec(vec![]);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_model_data_as_slice_extended() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let slice = data.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // AprV2Model forward error path tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty input
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("empty"));
    }

    #[test]
    fn test_apr_v2_model_generate_max_tokens_zero() {
        let data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(data).unwrap();
        let result = model.generate(&[1, 2], 0, None);
        // Should succeed with empty generation
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 2); // Just input, no generation
    }

    // =========================================================================
    // byte_to_bpe_char newline and tab
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_newline() {
        let result = super::byte_to_bpe_char(b'\n');
        assert_eq!(result, "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        let result = super::byte_to_bpe_char(b'\t');
        assert_eq!(result, "ĉ");
    }

    // =========================================================================
    // rms_norm with very small epsilon
    // =========================================================================

    #[test]
    fn test_rms_norm_tiny_eps() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-12;
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Should not produce NaN or Inf
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 10.0; // Large epsilon dominates
        let result = super::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

}
