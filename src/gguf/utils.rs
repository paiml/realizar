//! GGUF utility functions
//!
//! Private helpers used across GGUF modules for parsing, tensor manipulation,
//! and inference operations.

use std::sync::OnceLock;

// ============================================================================
// Verbose mode helper
// ============================================================================

/// Check if verbose mode is enabled (REALIZAR_VERBOSE=1)
/// Default is quiet - only errors are printed
pub(crate) fn verbose() -> bool {
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("REALIZAR_VERBOSE").is_ok())
}

// ============================================================================
// GPT-2 BPE Unicode utilities
// ============================================================================

/// Convert GPT-2 style byte-level BPE unicode character back to raw byte.
///
/// GPT-2's byte-level BPE uses a mapping where:
/// - Printable ASCII (0x21-0x7E) and Latin-1 (0xA1-0xAC, 0xAE-0xFF) map to themselves
/// - Other bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to U+0100-U+0143
///
/// This function returns the original byte value for a GPT-2 BPE token character.
#[inline]
pub(crate) fn gpt2_unicode_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;

    // Special encoded bytes: U+0100 to U+0143 map back to non-printable/special bytes
    if (0x0100..=0x0143).contains(&cp) {
        let offset = (cp - 0x0100) as u8;
        // The special bytes in order: 0x00-0x20 (0-32), then 0x7F (33), then 0x80-0xA0 (34-66), then 0xAD (67)
        let byte = if offset <= 32 {
            offset // 0x00-0x20
        } else if offset == 33 {
            0x7F // DEL
        } else if offset <= 66 {
            0x80 + (offset - 34) // 0x80-0xA0
        } else {
            0xAD // Soft hyphen
        };
        Some(byte)
    } else if cp <= 0xFF {
        // Direct mapping for printable chars
        Some(cp as u8)
    } else {
        None
    }
}

/// Decode a GPT-2 style byte-level BPE token to raw bytes.
///
/// Each character in the token may represent either a direct byte (printable ASCII/Latin-1)
/// or an encoded byte (using Unicode codepoints U+0100-U+0143).
pub(crate) fn decode_gpt2_token_to_bytes(token: &str) -> Vec<u8> {
    token.chars().filter_map(gpt2_unicode_to_byte).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbose_default_false() {
        // Unless REALIZAR_VERBOSE is set, should be false
        // Note: This may be true if env var is set during testing
        let _ = verbose(); // Just verify it doesn't panic
    }

    #[test]
    fn test_gpt2_unicode_to_byte_printable() {
        // Printable ASCII maps to itself
        assert_eq!(gpt2_unicode_to_byte('A'), Some(0x41));
        assert_eq!(gpt2_unicode_to_byte('z'), Some(0x7A));
        assert_eq!(gpt2_unicode_to_byte('!'), Some(0x21));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_special() {
        // Special encoded bytes U+0100-U+0143
        assert_eq!(gpt2_unicode_to_byte('\u{0100}'), Some(0x00)); // NUL
        assert_eq!(gpt2_unicode_to_byte('\u{0120}'), Some(0x20)); // Space (offset 32)
        assert_eq!(gpt2_unicode_to_byte('\u{0121}'), Some(0x7F)); // DEL (offset 33)
    }

    #[test]
    fn test_decode_gpt2_token() {
        let token = "Hello";
        let bytes = decode_gpt2_token_to_bytes(token);
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_decode_gpt2_token_with_special() {
        // Token with encoded space character
        let bytes = decode_gpt2_token_to_bytes("A\u{0120}B");
        assert_eq!(bytes, vec![0x41, 0x20, 0x42]); // A, space, B
    }
}
