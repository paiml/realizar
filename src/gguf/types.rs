//! GGUF type definitions and constants
//!
//! Core types used throughout the GGUF parser and model loading.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for testing and documentation.

// Re-export from monolith during migration
pub use super::monolith::{
    AttentionBuffer, GGUFHeader, GGUFModel, GGUFValue, HiddenBuffer, TensorInfo, TokenBuffer,
    ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE, GGUF_ALIGNMENT,
    GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_0,
    GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3, HIDDEN_BUFFER_INLINE_CAP,
    TOKEN_BUFFER_INLINE_CAP,
};

#[cfg(test)]
mod tests {
    use super::*;

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
