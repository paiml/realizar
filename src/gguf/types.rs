//! GGUF Types and Constants
//!
//! Core constants for GGUF format parsing.
//! Note: GGUFValue enum remains in monolith during migration.

// Re-export from monolith - these will be migrated here incrementally
// For now, this module only adds tests for the constants

#[cfg(test)]
mod tests {
    use crate::gguf::{
        ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE, GGUF_MAGIC,
        GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
        GGUF_TYPE_Q8_0, GGUF_VERSION_V3, HIDDEN_BUFFER_INLINE_CAP, TOKEN_BUFFER_INLINE_CAP,
    };

    #[test]
    fn test_magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
    }

    #[test]
    fn test_quantization_constants() {
        assert_eq!(GGUF_TYPE_F32, 0);
        assert_eq!(GGUF_TYPE_F16, 1);
        assert_eq!(GGUF_TYPE_Q4_0, 2);
        assert_eq!(GGUF_TYPE_Q8_0, 8);
        assert_eq!(GGUF_TYPE_Q4_K, 12);
        assert_eq!(GGUF_TYPE_Q6_K, 14);
    }

    #[test]
    fn test_buffer_constants() {
        assert_eq!(TOKEN_BUFFER_INLINE_CAP, 32);
        assert_eq!(ATTENTION_BUFFER_INLINE_CAP, 64);
        assert_eq!(HIDDEN_BUFFER_INLINE_CAP, 128);
    }

    #[test]
    fn test_buffer_watermarks() {
        assert_eq!(BUFFER_LW_SIZE, 1024);
        assert_eq!(BUFFER_HW_SIZE, 8 * 1024);
        assert_eq!(BUFFER_MAX_SIZE, 32 * 1024);
    }

    #[test]
    fn test_version_constant() {
        assert_eq!(GGUF_VERSION_V3, 3);
    }
}
