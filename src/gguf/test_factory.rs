//! GGUF Test Factory - Synthesizes valid GGUF files in memory
//!
//! This module provides `GGUFBuilder` for creating valid GGUF v3 files
//! without needing real model files. Essential for testing transformer
//! loading code that requires properly formatted binary data.
//!
//! # Example
//!
//! ```ignore
//! let data = GGUFBuilder::new()
//!     .architecture("llama")
//!     .hidden_dim(64)
//!     .num_layers(1)
//!     .add_f32_tensor("token_embd.weight", &[100, 64], &embedding_data)
//!     .add_q4_k_tensor("blk.0.attn_q.weight", &[64, 64], &q4k_data)
//!     .build();
//!
//! let model = GGUFModel::from_bytes(&data)?;
//! ```

use super::types::{
    GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

/// Builder for creating valid GGUF v3 files in memory
pub struct GGUFBuilder {
    /// Metadata key-value pairs (key, type, value_bytes)
    metadata: Vec<(String, u32, Vec<u8>)>,
    /// Tensor info entries (name, dims, qtype, data)
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for GGUFBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GGUFBuilder {
    /// Create a new GGUF builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    // =========================================================================
    // Metadata Helpers
    // =========================================================================

    /// Add a string metadata value
    #[must_use]
    pub fn add_string(mut self, key: &str, value: &str) -> Self {
        let mut bytes = Vec::new();
        // String: u64 length + UTF-8 bytes
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
        self.metadata.push((key.to_string(), 8, bytes)); // type 8 = string
        self
    }

    /// Add a u32 metadata value
    #[must_use]
    pub fn add_u32(mut self, key: &str, value: u32) -> Self {
        self.metadata
            .push((key.to_string(), 4, value.to_le_bytes().to_vec())); // type 4 = u32
        self
    }

    /// Add a f32 metadata value
    #[must_use]
    pub fn add_f32(mut self, key: &str, value: f32) -> Self {
        self.metadata
            .push((key.to_string(), 6, value.to_le_bytes().to_vec())); // type 6 = f32
        self
    }

    /// Set architecture (shorthand for general.architecture)
    #[must_use]
    pub fn architecture(self, arch: &str) -> Self {
        self.add_string("general.architecture", arch)
    }

    /// Set hidden dimension (embedding length)
    #[must_use]
    pub fn hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(&format!("{}.embedding_length", arch), dim)
    }

    /// Set number of layers (block count)
    #[must_use]
    pub fn num_layers(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{}.block_count", arch), count)
    }

    /// Set number of attention heads
    #[must_use]
    pub fn num_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{}.attention.head_count", arch), count)
    }

    /// Set number of KV heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{}.attention.head_count_kv", arch), count)
    }

    /// Set context length
    #[must_use]
    pub fn context_length(self, arch: &str, len: u32) -> Self {
        self.add_u32(&format!("{}.context_length", arch), len)
    }

    /// Set RoPE frequency base
    #[must_use]
    pub fn rope_freq_base(self, arch: &str, base: f32) -> Self {
        self.add_f32(&format!("{}.rope.freq_base", arch), base)
    }

    /// Set RMS epsilon
    #[must_use]
    pub fn rms_epsilon(self, arch: &str, eps: f32) -> Self {
        self.add_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch), eps)
    }

    /// Set feed-forward hidden dimension
    #[must_use]
    pub fn ffn_hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(&format!("{}.feed_forward_length", arch), dim)
    }

    /// Set vocab size (for completeness)
    #[must_use]
    pub fn vocab_size(self, _arch: &str, size: u32) -> Self {
        // Vocab size is typically inferred from token_embd.weight shape
        // But we can store it in metadata if needed
        self.add_u32("tokenizer.ggml.tokens.size", size)
    }

    // =========================================================================
    // Tensor Helpers
    // =========================================================================

    /// Add an F32 tensor
    #[must_use]
    pub fn add_f32_tensor(mut self, name: &str, dims: &[u64], data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_F32, bytes));
        self
    }

    /// Add a Q4_0 tensor (18 bytes per 32 elements)
    #[must_use]
    pub fn add_q4_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_Q4_0, data.to_vec()));
        self
    }

    /// Add a Q8_0 tensor (34 bytes per 32 elements)
    #[must_use]
    pub fn add_q8_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_Q8_0, data.to_vec()));
        self
    }

    /// Add a Q4_K tensor (144 bytes per 256 elements)
    #[must_use]
    pub fn add_q4_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_Q4_K, data.to_vec()));
        self
    }

    /// Add a Q5_K tensor (176 bytes per 256 elements)
    #[must_use]
    pub fn add_q5_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_Q5_K, data.to_vec()));
        self
    }

    /// Add a Q6_K tensor (210 bytes per 256 elements)
    #[must_use]
    pub fn add_q6_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_Q6_K, data.to_vec()));
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the GGUF file as a byte vector
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // Metadata
        for (key, value_type, value_bytes) in &self.metadata {
            // Key string: u64 length + UTF-8 bytes
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            // Value type
            data.extend_from_slice(&value_type.to_le_bytes());
            // Value bytes
            data.extend_from_slice(value_bytes);
        }

        // Tensor info
        let mut tensor_data_offset = 0u64;
        for (name, dims, qtype, tensor_bytes) in &self.tensors {
            // Name string
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());

            // n_dims
            data.extend_from_slice(&(dims.len() as u32).to_le_bytes());

            // Dimensions (reversed for GGML order)
            for dim in dims.iter().rev() {
                data.extend_from_slice(&dim.to_le_bytes());
            }

            // Quantization type
            data.extend_from_slice(&qtype.to_le_bytes());

            // Offset (relative to tensor data start)
            data.extend_from_slice(&tensor_data_offset.to_le_bytes());

            tensor_data_offset += tensor_bytes.len() as u64;
        }

        // Align to GGUF_ALIGNMENT (32 bytes)
        let current_len = data.len();
        let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
        data.resize(aligned, 0);

        // Tensor data
        for (_, _, _, tensor_bytes) in &self.tensors {
            data.extend_from_slice(tensor_bytes);
        }

        data
    }
}

// =============================================================================
// Helper Functions for Creating Quantized Data
// =============================================================================

/// Create valid Q4_0 data for a tensor with given dimensions
/// Q4_0: 18 bytes per 32 elements (2 f16 scale + 16 bytes quants)
#[must_use]
pub fn create_q4_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 18);

    for _ in 0..num_blocks {
        // f16 scale = 0.1
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 16 bytes of quants (mid-range values)
        data.extend([0x88u8; 16]);
    }

    data
}

/// Create valid Q8_0 data for a tensor with given dimensions
/// Q8_0: 34 bytes per 32 elements (2 f16 scale + 32 i8 quants)
#[must_use]
pub fn create_q8_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 34);

    for _ in 0..num_blocks {
        // f16 scale = 0.1
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 32 i8 quants (zeros)
        data.extend([0i8 as u8; 32]);
    }

    data
}

/// Create valid Q4_K data for a tensor with given dimensions
/// Q4_K: 144 bytes per 256 elements
#[must_use]
pub fn create_q4_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 144]
}

/// Create valid Q5_K data for a tensor with given dimensions
/// Q5_K: 176 bytes per 256 elements
#[must_use]
pub fn create_q5_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 176]
}

/// Create valid Q6_K data for a tensor with given dimensions
/// Q6_K: 210 bytes per 256 elements
#[must_use]
pub fn create_q6_k_data(num_elements: usize) -> Vec<u8> {
    let num_super_blocks = num_elements.div_ceil(256);
    vec![0u8; num_super_blocks * 210]
}

/// Create F32 embedding data (small random-ish values)
#[must_use]
pub fn create_f32_embedding_data(vocab_size: usize, hidden_dim: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(vocab_size * hidden_dim);
    for i in 0..(vocab_size * hidden_dim) {
        // Pseudo-random but deterministic values
        let val = ((i % 1000) as f32 - 500.0) / 5000.0;
        data.push(val);
    }
    data
}

/// Create F32 norm weights (typically ~1.0)
#[must_use]
pub fn create_f32_norm_weights(dim: usize) -> Vec<f32> {
    vec![1.0f32; dim]
}

// =============================================================================
// Complete Model Builder
// =============================================================================

/// Build a minimal valid LLaMA-style GGUF model
///
/// This creates a complete model with:
/// - Token embeddings (F32)
/// - One transformer layer with Q4_K weights
/// - Output norm (F32)
/// - LM head (tied to token embeddings)
#[must_use]
pub fn build_minimal_llama_gguf(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<u8> {
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    // Q4_K weights for layer 0
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    GGUFBuilder::new()
        // Metadata
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", num_heads as u32)
        .num_kv_heads("llama", num_kv_heads as u32)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        // Token embedding
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        // Layer 0 attention
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        // Layer 0 FFN
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        // Output norm and head
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        // Note: LM head often tied to token_embd, so we don't add output.weight
        // The loader will fallback to token_embd.weight
        .build()
}

/// Build a minimal Phi-2 style GGUF model (fused QKV)
#[must_use]
pub fn build_minimal_phi2_gguf(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
) -> Vec<u8> {
    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    // Fused QKV: hidden -> 3 * hidden
    let qkv_out_dim = 3 * hidden_dim;
    let qkv_data = create_q4_k_data(hidden_dim * qkv_out_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);

    GGUFBuilder::new()
        // Metadata
        .architecture("phi2")
        .hidden_dim("phi2", hidden_dim as u32)
        .num_layers("phi2", 1)
        .num_heads("phi2", num_heads as u32)
        .num_kv_heads("phi2", num_heads as u32) // MHA, not GQA
        .context_length("phi2", 256)
        .rope_freq_base("phi2", 10000.0)
        .rms_epsilon("phi2", 1e-5)
        // Token embedding
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        // Layer 0 attention (fused QKV)
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_qkv.weight",
            &[hidden_dim as u64, qkv_out_dim as u64],
            &qkv_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        // Layer 0 FFN (no gate for Phi-2 style GELU)
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        // Output
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUFModel;

    #[test]
    fn test_gguf_builder_empty() {
        let data = GGUFBuilder::new().build();

        // Should have valid header
        assert!(data.len() >= 24); // magic + version + 2 counts

        let model = GGUFModel::from_bytes(&data).expect("Should parse empty GGUF");
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, GGUF_VERSION_V3);
        assert_eq!(model.metadata.len(), 0);
        assert_eq!(model.tensors.len(), 0);
    }

    #[test]
    fn test_gguf_builder_metadata_only() {
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_u32("test.value", 42)
            .add_f32("test.float", 3.14)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.metadata.len(), 3);
        assert_eq!(model.architecture(), Some("llama"));
    }

    #[test]
    fn test_gguf_builder_with_tensor() {
        let data = GGUFBuilder::new()
            .add_f32_tensor("test.weight", &[4, 8], &vec![0.0f32; 32])
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "test.weight");
        assert_eq!(model.tensors[0].n_dims, 2);
    }

    #[test]
    fn test_gguf_builder_q4_k_tensor() {
        let q4k_data = create_q4_k_data(256);
        let data = GGUFBuilder::new()
            .add_q4_k_tensor("layer.weight", &[256], &q4k_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q4_K);
    }

    #[test]
    fn test_minimal_llama_model() {
        let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);

        let model = GGUFModel::from_bytes(&data).expect("Should parse minimal LLaMA");

        assert_eq!(model.architecture(), Some("llama"));
        assert_eq!(model.embedding_dim(), Some(64));
        assert_eq!(model.num_layers(), Some(1));
        assert_eq!(model.num_heads(), Some(4));

        // Should have all expected tensors
        let tensor_names: Vec<_> = model.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(tensor_names.contains(&"token_embd.weight"));
        assert!(tensor_names.contains(&"blk.0.attn_q.weight"));
        assert!(tensor_names.contains(&"blk.0.ffn_up.weight"));
        assert!(tensor_names.contains(&"output_norm.weight"));
    }
}
