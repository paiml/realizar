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

    #[test]
    fn test_gpt2_unicode_to_byte_null() {
        // NUL character (0x00) encoded as U+0100
        assert_eq!(gpt2_unicode_to_byte('\u{0100}'), Some(0x00));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_tab() {
        // TAB (0x09) encoded as U+0109
        assert_eq!(gpt2_unicode_to_byte('\u{0109}'), Some(0x09));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_newline() {
        // Newline (0x0A) encoded as U+010A
        assert_eq!(gpt2_unicode_to_byte('\u{010A}'), Some(0x0A));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_carriage_return() {
        // CR (0x0D) encoded as U+010D
        assert_eq!(gpt2_unicode_to_byte('\u{010D}'), Some(0x0D));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_latin1() {
        // Latin-1 characters (0xA1-0xAC, 0xAE-0xFF) map to themselves
        assert_eq!(gpt2_unicode_to_byte('¡'), Some(0xA1)); // Inverted exclamation
        assert_eq!(gpt2_unicode_to_byte('ñ'), Some(0xF1)); // N with tilde
        assert_eq!(gpt2_unicode_to_byte('ÿ'), Some(0xFF)); // Y with diaeresis
    }

    #[test]
    fn test_gpt2_unicode_to_byte_extended_special() {
        // Extended special range (0x80-0xA0)
        assert_eq!(gpt2_unicode_to_byte('\u{0122}'), Some(0x80)); // offset 34 -> 0x80
        assert_eq!(gpt2_unicode_to_byte('\u{0142}'), Some(0xA0)); // offset 66 -> 0xA0
    }

    #[test]
    fn test_gpt2_unicode_to_byte_soft_hyphen() {
        // Soft hyphen (0xAD) encoded as U+0143
        assert_eq!(gpt2_unicode_to_byte('\u{0143}'), Some(0xAD));
    }

    #[test]
    fn test_gpt2_unicode_to_byte_out_of_range() {
        // Characters outside the valid range return None
        assert_eq!(gpt2_unicode_to_byte('\u{0200}'), None);
        assert_eq!(gpt2_unicode_to_byte('\u{1000}'), None);
        assert_eq!(gpt2_unicode_to_byte('🎉'), None);
    }

    #[test]
    fn test_decode_gpt2_token_empty() {
        let bytes = decode_gpt2_token_to_bytes("");
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_decode_gpt2_token_numbers() {
        let bytes = decode_gpt2_token_to_bytes("12345");
        assert_eq!(bytes, b"12345");
    }

    #[test]
    fn test_decode_gpt2_token_punctuation() {
        let bytes = decode_gpt2_token_to_bytes("!@#$%");
        assert_eq!(bytes, b"!@#$%");
    }

    #[test]
    fn test_decode_gpt2_token_mixed() {
        // Mix of printable ASCII and encoded special chars
        let bytes = decode_gpt2_token_to_bytes("Hello\u{010A}World");
        assert_eq!(bytes, vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x0A, 0x57, 0x6F, 0x72, 0x6C, 0x64]);
    }

    #[test]
    fn test_decode_gpt2_token_filters_invalid() {
        // Characters that can't be decoded are filtered out
        let bytes = decode_gpt2_token_to_bytes("A🎉B");
        assert_eq!(bytes, vec![0x41, 0x42]); // A, B - emoji filtered
    }

    #[test]
    fn test_gpt2_unicode_to_byte_all_printable_ascii() {
        // Test all printable ASCII (0x21-0x7E)
        for byte in 0x21u8..=0x7E {
            let c = byte as char;
            assert_eq!(gpt2_unicode_to_byte(c), Some(byte));
        }
    }

    #[test]
    fn test_verbose_returns_consistent() {
        // Calling verbose multiple times should return the same value
        let first = verbose();
        let second = verbose();
        assert_eq!(first, second);
    }
}
