//! Unified Model Format Detection and Loading
//!
//! Per spec §3: Format Support Matrix - auto-detect APR, GGUF, SafeTensors from magic bytes.
//!
//! ## Jidoka (Built-in Quality)
//!
//! - CRC32 verification for APR format
//! - Header size validation for SafeTensors (DOS protection)
//! - Magic byte validation for GGUF
//!
//! ## Supported Formats
//!
//! | Format | Magic | Extension |
//! |--------|-------|-----------|
//! | APR    | `APR\0` | `.apr` |
//! | GGUF   | `GGUF` | `.gguf` |
//! | SafeTensors | (u64 header size) | `.safetensors` |

use std::path::Path;

/// Detected model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Aprender native format (first-class support)
    Apr,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// SafeTensors format (HuggingFace compatible)
    SafeTensors,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Apr => write!(f, "APR"),
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

/// Errors during format detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// Data too short for format detection (need at least 8 bytes)
    TooShort {
        /// Actual length
        len: usize,
    },
    /// Unknown format (no magic bytes matched)
    UnknownFormat,
    /// SafeTensors header too large (DOS protection per spec §7.1)
    HeaderTooLarge {
        /// Header size in bytes
        size: u64,
    },
    /// File extension doesn't match detected format
    ExtensionMismatch {
        /// Detected format
        detected: ModelFormat,
        /// Extension from filename
        extension: String,
    },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { len } => {
                write!(
                    f,
                    "Data too short for format detection: {len} bytes (need 8)"
                )
            },
            Self::UnknownFormat => write!(f, "Unknown model format (no magic bytes matched)"),
            Self::HeaderTooLarge { size } => write!(
                f,
                "SafeTensors header too large: {size} bytes (max 100MB for DOS protection)"
            ),
            Self::ExtensionMismatch {
                detected,
                extension,
            } => {
                write!(
                    f,
                    "Extension mismatch: detected {detected} but file has extension .{extension}"
                )
            },
        }
    }
}

impl std::error::Error for FormatError {}

/// APR format magic bytes
///
/// ONE format. No versioning. Period.
pub const APR_MAGIC: &[u8; 4] = b"APR\0";

/// GGUF format magic bytes
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Maximum SafeTensors header size (100MB for DOS protection per spec §7.1)
pub const MAX_SAFETENSORS_HEADER: u64 = 100_000_000;

/// Detect model format from magic bytes (Jidoka: fail-fast)
///
/// Per spec §3.2: Format Detection
///
/// # Arguments
///
/// * `data` - First 8+ bytes of the model file
///
/// # Returns
///
/// Detected format or error
///
/// # Errors
///
/// Returns error if:
/// - Data is too short (<8 bytes)
/// - No known magic bytes detected
/// - SafeTensors header size exceeds limit (DOS protection)
///
/// # Example
///
/// ```
/// use realizar::format::{detect_format, ModelFormat};
///
/// // APR format
/// let apr_data = b"APR\0xxxxxxxxxxxx";
/// assert_eq!(detect_format(apr_data).expect("test"), ModelFormat::Apr);
///
/// // GGUF format
/// let gguf_data = b"GGUFxxxxxxxxxxxx";
/// assert_eq!(detect_format(gguf_data).expect("test"), ModelFormat::Gguf);
/// ```
pub fn detect_format(data: &[u8]) -> Result<ModelFormat, FormatError> {
    if data.len() < 8 {
        return Err(FormatError::TooShort { len: data.len() });
    }

    // Check APR magic - ONE format, no versioning
    if &data[0..4] == APR_MAGIC {
        return Ok(ModelFormat::Apr);
    }

    // Check GGUF magic
    if &data[0..4] == GGUF_MAGIC {
        return Ok(ModelFormat::Gguf);
    }

    // SafeTensors: first 8 bytes are header size (little-endian u64)
    // If it's a reasonable size, assume SafeTensors
    let header_size = u64::from_le_bytes(data[0..8].try_into().expect("slice is exactly 8 bytes"));

    // SafeTensors header should be reasonable size
    // Very large values indicate this isn't SafeTensors
    if header_size < MAX_SAFETENSORS_HEADER && header_size > 0 {
        return Ok(ModelFormat::SafeTensors);
    }

    // Check if header size looks like SafeTensors but is too large
    if header_size >= MAX_SAFETENSORS_HEADER {
        return Err(FormatError::HeaderTooLarge { size: header_size });
    }

    Err(FormatError::UnknownFormat)
}

/// Detect format from file path (using extension as hint, then verify magic)
///
/// # Arguments
///
/// * `path` - Path to model file
///
/// # Returns
///
/// Detected format (verified against magic bytes if data provided)
///
/// # Errors
///
/// Returns `FormatError::UnknownFormat` if extension is not recognized.
///
/// # Example
///
/// ```
/// use realizar::format::{detect_format_from_path, ModelFormat};
/// use std::path::Path;
///
/// assert_eq!(
///     detect_format_from_path(Path::new("model.apr")).expect("test"),
///     ModelFormat::Apr
/// );
/// ```
pub fn detect_format_from_path(path: &Path) -> Result<ModelFormat, FormatError> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "apr" => Ok(ModelFormat::Apr),
        "gguf" => Ok(ModelFormat::Gguf),
        "safetensors" => Ok(ModelFormat::SafeTensors),
        _ => Err(FormatError::UnknownFormat),
    }
}

/// Detect format from path and verify against data magic bytes
///
/// Per Jidoka: stop immediately if extension doesn't match magic
///
/// # Arguments
///
/// * `path` - Path to model file
/// * `data` - First 8+ bytes of model data
///
/// # Returns
///
/// Verified format or error if mismatch
///
/// # Errors
///
/// Returns error if:
/// - Format cannot be detected from magic bytes
/// - File extension doesn't match detected format
pub fn detect_and_verify_format(path: &Path, data: &[u8]) -> Result<ModelFormat, FormatError> {
    let from_data = detect_format(data)?;
    let from_path = detect_format_from_path(path);

    // If path detection succeeded, verify it matches data
    if let Ok(path_format) = from_path {
        if path_format != from_data {
            return Err(FormatError::ExtensionMismatch {
                detected: from_data,
                extension: path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
    }

    // Data-based detection is authoritative
    Ok(from_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== EXTREME TDD: Format Detection Tests =====

    #[test]
    fn test_detect_apr_format() {
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_gguf_format() {
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_safetensors_format() {
        // SafeTensors: first 8 bytes are header size (little-endian)
        // A reasonable header size like 1000 bytes
        let mut data = vec![0u8; 16];
        let header_size: u64 = 1000;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_too_short() {
        let data = b"APR"; // Only 3 bytes
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 3 })));
    }

    #[test]
    fn test_detect_format_empty() {
        let data: &[u8] = &[];
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 0 })));
    }

    #[test]
    fn test_detect_safetensors_header_too_large() {
        // Header size > 100MB should fail (DOS protection)
        let mut data = vec![0u8; 16];
        let header_size: u64 = 200_000_000; // 200MB
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        assert!(matches!(
            result,
            Err(FormatError::HeaderTooLarge { size: 200_000_000 })
        ));
    }

    #[test]
    fn test_detect_unknown_format() {
        // Random bytes that don't match any format
        // Zero header size means not SafeTensors either
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00xxxx";
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_apr() {
        let path = Path::new("model.apr");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_format_from_path_gguf() {
        let path = Path::new("llama-7b-q4.gguf");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_safetensors() {
        let path = Path::new("model.safetensors");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_from_path_unknown() {
        let path = Path::new("model.bin");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_uppercase() {
        // Extension comparison should be case-insensitive
        let path = Path::new("MODEL.APR");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_and_verify_format_match() {
        let path = Path::new("model.apr");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_detect_and_verify_format_mismatch() {
        let path = Path::new("model.apr"); // Says APR
        let data = b"GGUFxxxxxxxxxxxxxxxx"; // But data is GGUF
        let result = detect_and_verify_format(path, data);
        assert!(matches!(
            result,
            Err(FormatError::ExtensionMismatch {
                detected: ModelFormat::Gguf,
                ..
            })
        ));
    }

    #[test]
    fn test_detect_and_verify_unknown_extension_ok() {
        // Unknown extension but valid magic should work
        let path = Path::new("model.bin");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Apr
        );
    }

    #[test]
    fn test_model_format_display() {
        assert_eq!(format!("{}", ModelFormat::Apr), "APR");
        assert_eq!(format!("{}", ModelFormat::Gguf), "GGUF");
        assert_eq!(format!("{}", ModelFormat::SafeTensors), "SafeTensors");
    }

    #[test]
    fn test_format_error_display() {
        let err = FormatError::TooShort { len: 5 };
        assert!(err.to_string().contains("5 bytes"));

        let err = FormatError::UnknownFormat;
        assert!(err.to_string().contains("Unknown"));

        let err = FormatError::HeaderTooLarge { size: 999 };
        assert!(err.to_string().contains("999 bytes"));

        let err = FormatError::ExtensionMismatch {
            detected: ModelFormat::Gguf,
            extension: "apr".to_string(),
        };
        assert!(err.to_string().contains("GGUF"));
        assert!(err.to_string().contains(".apr"));
    }

    #[test]
    fn test_magic_constants() {
        assert_eq!(APR_MAGIC, b"APR\0");
        assert_eq!(GGUF_MAGIC, b"GGUF");
        assert_eq!(MAX_SAFETENSORS_HEADER, 100_000_000);
    }

    // ===== Property-based edge cases =====

    #[test]
    fn test_exactly_8_bytes_safetensors() {
        // Exactly 8 bytes with valid header size
        let header_size: u64 = 500;
        let data = header_size.to_le_bytes();
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_apr_with_trailing_data() {
        // APR magic followed by lots of other data
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&[0u8; 1000]);
        assert_eq!(detect_format(&data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_gguf_with_trailing_data() {
        // GGUF magic followed by lots of other data
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[0u8; 1000]);
        assert_eq!(detect_format(&data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_safetensors_boundary_header_size() {
        // Just under the limit
        let mut data = vec![0u8; 16];
        let header_size: u64 = MAX_SAFETENSORS_HEADER - 1;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_format(&data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_safetensors_exactly_at_limit() {
        // Exactly at limit should fail
        let mut data = vec![0u8; 16];
        let header_size: u64 = MAX_SAFETENSORS_HEADER;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        assert!(matches!(result, Err(FormatError::HeaderTooLarge { .. })));
    }
}
