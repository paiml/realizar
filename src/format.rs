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

/// APR format magic bytes (first 3 bytes, 4th is version)
///
/// APR v1: `APR1` (0x41505231)
/// APR v2: `APR2` (0x41505232)
/// Legacy: `APR\0` (0x41505200)
pub const APR_MAGIC: &[u8; 3] = b"APR";

/// GGUF format magic bytes
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Maximum SafeTensors header size (100MB for DOS protection per spec §7.1)
pub const MAX_SAFETENSORS_HEADER: u64 = 100_000_000;

/// Valid APR version bytes
const APR_VERSIONS: [u8; 4] = [b'N', b'1', b'2', 0];

/// Try to detect APR format from magic bytes
#[inline]
fn try_detect_apr(data: &[u8]) -> Option<ModelFormat> {
    if data.len() >= 4 && &data[0..3] == APR_MAGIC && APR_VERSIONS.contains(&data[3]) {
        return Some(ModelFormat::Apr);
    }
    None
}

/// Try to detect GGUF format from magic bytes
#[inline]
fn try_detect_gguf(data: &[u8]) -> Option<ModelFormat> {
    if data.len() >= 4 && &data[0..4] == GGUF_MAGIC {
        return Some(ModelFormat::Gguf);
    }
    None
}

/// Try to detect SafeTensors format from header size
#[inline]
fn try_detect_safetensors(data: &[u8]) -> Result<Option<ModelFormat>, FormatError> {
    let header_size = u64::from_le_bytes(data[0..8].try_into().expect("slice is exactly 8 bytes"));
    if header_size > 0 && header_size < MAX_SAFETENSORS_HEADER {
        return Ok(Some(ModelFormat::SafeTensors));
    }
    if header_size >= MAX_SAFETENSORS_HEADER {
        return Err(FormatError::HeaderTooLarge { size: header_size });
    }
    Ok(None)
}

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

    // Try each format in order of specificity
    if let Some(format) = try_detect_apr(data) {
        return Ok(format);
    }
    if let Some(format) = try_detect_gguf(data) {
        return Ok(format);
    }
    if let Some(format) = try_detect_safetensors(data)? {
        return Ok(format);
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
    fn test_detect_apr_format_legacy() {
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_apr_format_v1() {
        let data = b"APR1xxxxxxxxxxxxxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_apr_format_v2() {
        let data = b"APR2xxxxxxxxxxxxxxxx";
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
        // APR_MAGIC is now 3 bytes, version in 4th byte
        assert_eq!(APR_MAGIC, b"APR");
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

    // ===== T-COV-95 Phase 60: Extended format detection coverage =====

    // ----- APR version byte coverage -----

    #[test]
    fn test_detect_apr_format_version_n() {
        // APR_VERSIONS includes b'N'
        let data = b"APRNxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_format(data).expect("APRN should be valid APR"),
            ModelFormat::Apr,
            "Version byte 'N' should be recognized as valid APR"
        );
    }

    #[test]
    fn test_detect_apr_invalid_version_byte() {
        // Version byte 'X' is NOT in APR_VERSIONS [b'N', b'1', b'2', 0]
        let data = b"APRXxxxxxxxxxxxxxxxx".to_vec();
        // This has "APR" prefix but invalid version, should NOT be detected as APR
        // It will fall through to GGUF check (fails), then SafeTensors check
        let result = detect_format(&data);
        assert_ne!(
            result.as_ref().ok().copied(),
            Some(ModelFormat::Apr),
            "Invalid version byte 'X' should not be recognized as APR"
        );
    }

    #[test]
    fn test_detect_apr_version_3_invalid() {
        // b'3' is NOT in APR_VERSIONS
        let data = b"APR3xxxxxxxxxxxxxxxx";
        let result = detect_format(data);
        assert_ne!(
            result.as_ref().ok().copied(),
            Some(ModelFormat::Apr),
            "Version byte '3' should not be recognized as APR"
        );
    }

    // ----- try_detect_apr edge cases -----

    #[test]
    fn test_try_detect_apr_exactly_4_bytes() {
        let data = b"APR\0";
        assert_eq!(try_detect_apr(data), Some(ModelFormat::Apr));
    }

    #[test]
    fn test_try_detect_apr_3_bytes_too_short() {
        let data = b"APR";
        assert_eq!(
            try_detect_apr(data),
            None,
            "3 bytes is not enough for APR detection"
        );
    }

    #[test]
    fn test_try_detect_apr_2_bytes() {
        let data = b"AP";
        assert_eq!(try_detect_apr(data), None);
    }

    #[test]
    fn test_try_detect_apr_empty() {
        let data: &[u8] = &[];
        assert_eq!(try_detect_apr(data), None);
    }

    #[test]
    fn test_try_detect_apr_wrong_magic() {
        let data = b"XPR\0xxxxxxxxx";
        assert_eq!(try_detect_apr(data), None);
    }

    // ----- try_detect_gguf edge cases -----

    #[test]
    fn test_try_detect_gguf_exactly_4_bytes() {
        let data = b"GGUF";
        assert_eq!(try_detect_gguf(data), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_try_detect_gguf_3_bytes_too_short() {
        let data = b"GGU";
        assert_eq!(
            try_detect_gguf(data),
            None,
            "3 bytes is not enough for GGUF detection"
        );
    }

    #[test]
    fn test_try_detect_gguf_empty() {
        let data: &[u8] = &[];
        assert_eq!(try_detect_gguf(data), None);
    }

    #[test]
    fn test_try_detect_gguf_wrong_magic() {
        let data = b"GGXFxxxxxxxx";
        assert_eq!(try_detect_gguf(data), None);
    }

    #[test]
    fn test_try_detect_gguf_lowercase_rejected() {
        let data = b"ggufxxxxxxxx";
        assert_eq!(
            try_detect_gguf(data),
            None,
            "lowercase gguf should not match"
        );
    }

    // ----- try_detect_safetensors edge cases -----

    #[test]
    fn test_try_detect_safetensors_header_size_zero() {
        let header_size: u64 = 0;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(
            result, None,
            "header_size=0 should not be detected as SafeTensors"
        );
    }

    #[test]
    fn test_try_detect_safetensors_header_size_one() {
        let header_size: u64 = 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(result, Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_try_detect_safetensors_header_just_below_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER - 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data).expect("should not error");
        assert_eq!(result, Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_try_detect_safetensors_header_at_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(
            result.is_err(),
            "header_size at MAX should return HeaderTooLarge error"
        );
        assert!(matches!(
            result.unwrap_err(),
            FormatError::HeaderTooLarge { size } if size == MAX_SAFETENSORS_HEADER
        ));
    }

    #[test]
    fn test_try_detect_safetensors_header_above_max() {
        let header_size: u64 = MAX_SAFETENSORS_HEADER + 1;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_detect_safetensors_u64_max() {
        let header_size: u64 = u64::MAX;
        let data = header_size.to_le_bytes();
        let result = try_detect_safetensors(&data);
        assert!(result.is_err(), "u64::MAX header should be rejected");
    }

    // ----- detect_format priority order -----

    #[test]
    fn test_detect_format_apr_takes_priority_over_safetensors() {
        // APR\0 could also be interpreted as a small header_size in little-endian
        // but APR detection should take priority
        let data = b"APR\0\x00\x00\x00\x00";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_gguf_takes_priority_over_safetensors() {
        // GGUF bytes could be a valid header_size, but GGUF should win
        let data = b"GGUFxxxx";
        assert_eq!(detect_format(data).expect("test"), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_7_bytes_too_short() {
        let data = b"APR\0xxx";
        let result = detect_format(data);
        assert!(matches!(result, Err(FormatError::TooShort { len: 7 })));
    }

    #[test]
    fn test_detect_format_exactly_8_bytes_unknown() {
        // 8 zero bytes: header_size=0, not APR, not GGUF -> UnknownFormat
        let data = [0u8; 8];
        let result = detect_format(&data);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    // ----- detect_format_from_path edge cases -----

    #[test]
    fn test_detect_format_from_path_no_extension() {
        let path = Path::new("model");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_dot_only() {
        let path = Path::new("model.");
        let result = detect_format_from_path(path);
        assert!(matches!(result, Err(FormatError::UnknownFormat)));
    }

    #[test]
    fn test_detect_format_from_path_double_extension() {
        // Only the last extension is checked
        let path = Path::new("model.tar.gguf");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_uppercase_gguf() {
        let path = Path::new("MODEL.GGUF");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_from_path_uppercase_safetensors() {
        let path = Path::new("MODEL.SAFETENSORS");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_from_path_mixed_case_apr() {
        let path = Path::new("model.ApR");
        assert_eq!(
            detect_format_from_path(path).expect("test"),
            ModelFormat::Apr
        );
    }

    // ----- detect_and_verify_format edge cases -----

    #[test]
    fn test_detect_and_verify_gguf_match() {
        let path = Path::new("model.gguf");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_and_verify_safetensors_match() {
        let path = Path::new("model.safetensors");
        let mut data = vec![0u8; 16];
        let header_size: u64 = 500;
        data[0..8].copy_from_slice(&header_size.to_le_bytes());
        assert_eq!(
            detect_and_verify_format(path, &data).expect("test"),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_and_verify_gguf_as_safetensors_mismatch() {
        let path = Path::new("model.safetensors");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
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
    fn test_detect_and_verify_apr_as_gguf_mismatch() {
        let path = Path::new("model.gguf");
        let data = b"APR\0xxxxxxxxxxxxxxxx";
        let result = detect_and_verify_format(path, data);
        assert!(matches!(
            result,
            Err(FormatError::ExtensionMismatch {
                detected: ModelFormat::Apr,
                ..
            })
        ));
    }

    #[test]
    fn test_detect_and_verify_no_extension_data_valid() {
        // No recognized extension, but data is valid GGUF
        let path = Path::new("model.xyz");
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        assert_eq!(
            detect_and_verify_format(path, data).expect("test"),
            ModelFormat::Gguf,
            "Unrecognized extension should not prevent data-based detection"
        );
    }

    #[test]
    fn test_detect_and_verify_data_too_short() {
        let path = Path::new("model.gguf");
        let data = b"GGU"; // Only 3 bytes
        let result = detect_and_verify_format(path, data);
        assert!(matches!(result, Err(FormatError::TooShort { .. })));
    }

    // ----- ModelFormat trait impls -----

    #[test]
    fn test_model_format_copy() {
        let fmt = ModelFormat::Apr;
        let copied = fmt; // Copy
        assert_eq!(fmt, copied);
    }

    #[test]
    fn test_model_format_clone() {
        let fmt = ModelFormat::SafeTensors;
        let cloned = fmt.clone();
        assert_eq!(fmt, cloned);
    }

    #[test]
    fn test_model_format_debug() {
        let debug = format!("{:?}", ModelFormat::Gguf);
        assert_eq!(debug, "Gguf");
    }

    #[test]
    fn test_model_format_eq_all_variants() {
        assert_eq!(ModelFormat::Apr, ModelFormat::Apr);
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::Apr, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Apr, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
    }

    // ----- FormatError trait impls -----

    #[test]
    fn test_format_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(FormatError::UnknownFormat);
        assert!(err.to_string().contains("Unknown"));
    }

    #[test]
    fn test_format_error_debug_all_variants() {
        let too_short = format!("{:?}", FormatError::TooShort { len: 3 });
        assert!(too_short.contains("TooShort"));
        assert!(too_short.contains("3"));

        let unknown = format!("{:?}", FormatError::UnknownFormat);
        assert!(unknown.contains("UnknownFormat"));

        let too_large = format!("{:?}", FormatError::HeaderTooLarge { size: 999 });
        assert!(too_large.contains("HeaderTooLarge"));
        assert!(too_large.contains("999"));

        let mismatch = format!(
            "{:?}",
            FormatError::ExtensionMismatch {
                detected: ModelFormat::Apr,
                extension: "gguf".to_string(),
            }
        );
        assert!(mismatch.contains("ExtensionMismatch"));
        assert!(mismatch.contains("Apr"));
        assert!(mismatch.contains("gguf"));
    }

    #[test]
    fn test_format_error_clone_all_variants() {
        let errors: Vec<FormatError> = vec![
            FormatError::TooShort { len: 5 },
            FormatError::UnknownFormat,
            FormatError::HeaderTooLarge { size: 42 },
            FormatError::ExtensionMismatch {
                detected: ModelFormat::Gguf,
                extension: "apr".to_string(),
            },
        ];
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(*err, cloned);
        }
    }

    #[test]
    fn test_format_error_eq() {
        assert_eq!(FormatError::UnknownFormat, FormatError::UnknownFormat);
        assert_eq!(
            FormatError::TooShort { len: 5 },
            FormatError::TooShort { len: 5 }
        );
        assert_ne!(
            FormatError::TooShort { len: 5 },
            FormatError::TooShort { len: 6 }
        );
        assert_ne!(FormatError::UnknownFormat, FormatError::TooShort { len: 0 });
    }

    // ----- FormatError Display messages are informative -----

    #[test]
    fn test_format_error_too_short_display_message() {
        let err = FormatError::TooShort { len: 0 };
        let msg = err.to_string();
        assert!(
            msg.contains("0 bytes"),
            "should show actual byte count: {}",
            msg
        );
        assert!(
            msg.contains("need 8"),
            "should show required minimum: {}",
            msg
        );
    }

    #[test]
    fn test_format_error_header_too_large_display_message() {
        let err = FormatError::HeaderTooLarge { size: 500_000_000 };
        let msg = err.to_string();
        assert!(
            msg.contains("500000000"),
            "should show actual size: {}",
            msg
        );
        assert!(msg.contains("100MB"), "should mention the limit: {}", msg);
    }

    #[test]
    fn test_format_error_extension_mismatch_display_message() {
        let err = FormatError::ExtensionMismatch {
            detected: ModelFormat::SafeTensors,
            extension: "gguf".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("SafeTensors"),
            "should show detected format: {}",
            msg
        );
        assert!(msg.contains(".gguf"), "should show file extension: {}", msg);
    }

    // ----- APR_VERSIONS coverage -----

    #[test]
    fn test_apr_versions_all_valid() {
        // Test all valid version bytes
        for &version in &APR_VERSIONS {
            let mut data = b"APR\0xxxxxxxxxxxxxxxx".to_vec();
            data[3] = version;
            assert_eq!(
                detect_format(&data).expect("valid APR version should be detected"),
                ModelFormat::Apr,
                "Version byte {} should be valid APR",
                version
            );
        }
    }

    #[test]
    fn test_apr_versions_invalid_adjacent() {
        // Test version bytes adjacent to valid ones that should NOT match
        let invalid_versions: &[u8] = &[b'0', b'3', b'4', b'M', b'O', 1, 255];
        for &version in invalid_versions {
            if APR_VERSIONS.contains(&version) {
                continue;
            }
            let mut data = b"APR\0xxxxxxxxxxxxxxxx".to_vec();
            data[3] = version;
            let result = detect_format(&data);
            if let Ok(fmt) = result {
                assert_ne!(
                    fmt,
                    ModelFormat::Apr,
                    "Invalid version byte {} should NOT be recognized as APR",
                    version
                );
            }
        }
    }

    // ----- SafeTensors header size boundary values -----

    #[test]
    fn test_safetensors_header_size_small_values() {
        for size in 1u64..=10 {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert_eq!(
                result.expect("small header sizes should be valid SafeTensors"),
                ModelFormat::SafeTensors,
                "Header size {} should be valid SafeTensors",
                size
            );
        }
    }

    #[test]
    fn test_safetensors_header_size_typical_values() {
        let typical_sizes: &[u64] = &[256, 1024, 4096, 65536, 1_000_000, 50_000_000];
        for &size in typical_sizes {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert_eq!(
                result.expect("typical header sizes should be valid SafeTensors"),
                ModelFormat::SafeTensors,
                "Header size {} should be valid SafeTensors",
                size
            );
        }
    }
}
