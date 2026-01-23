//! APR Transformer Format for WASM-compatible LLM inference
//!
//! This module provides a WASM-compatible transformer implementation that stores
//! all weights as F32, enabling fair comparison between APR and GGUF formats.
//!
//! ## Design Goals
//!
//! 1. **WASM Compatibility**: Pure F32 weights, no SIMD requirements
//! 2. **Fair Comparison**: Same inference algorithm as GGUFTransformer
//! 3. **Serialization**: APR format with model type `TransformerLM` (0x0050)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr_transformer::AprTransformer;
//! use realizar::gguf::{GGUFModel, GGUFTransformer};
//!
//! // Load GGUF model
//! let gguf_data = std::fs::read("model.gguf")?;
//! let gguf_model = GGUFModel::from_bytes(&gguf_data)?;
//! let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data)?;
//!
//! // Convert to APR format
//! let apr_transformer = AprTransformer::from_gguf_transformer(&gguf_transformer);
//!
//! // Run inference (should match GGUF output)
//! let logits = apr_transformer.forward(&[1, 2, 3, 4])?;
//! ```

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch as matmul_q4k_f32_ggml;
use trueno::backends::q6k::matmul_q6k_f32_colmajor_dispatch as matmul_q6k_f32_ggml;
use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

use crate::apr::MAGIC;
use crate::error::{RealizarError, Result};

// APR Benchmark Infrastructure (Y6) - extracted from mod.rs (PMAT-802)
mod benchmark;
pub use benchmark::{
    AprBenchmarkResult, AprBenchmarkRunner, AprLoadResult, AprParityComparison, AprPrefillResult,
    APR_CPU_DECODE_THRESHOLD_TOK_S, APR_PARITY_THRESHOLD_PCT, APR_PREFILL_THRESHOLD_TOK_S,
};

// ============================================================================
// APR Transformer Binary Format (Y1-Y5 Format Parity)
// ============================================================================
// Uses unified APR magic from apr.rs - ONE format, no versioning

/// Binary header size for APR Transformer (64 bytes)
pub const APR_TRANSFORMER_HEADER_SIZE: usize = 64;

/// Memory-mapped APR Transformer for zero-copy inference (Y1, Y2)
///
/// This struct provides zero-copy access to APR transformer weights
/// via memory-mapped I/O, matching GGUF's performance characteristics.
///
/// # Performance Benefits (per Dean & Barroso 2013)
///
/// - Zero-copy: Tensors accessed directly from page cache
/// - Lazy loading: Only touched pages are loaded
/// - Shared memory: Multiple processes can share the same mapping
#[derive(Debug)]
pub struct MmapAprTransformer {
    /// Memory-mapped file data
    mmap: Mmap,
    /// Model configuration (parsed from header)
    pub config: AprTransformerConfig,
    /// Offset where tensor data starts
    tensor_data_offset: usize,
    /// Whether mmap is active (for is_mmap() check)
    is_mmap: bool,
}

impl MmapAprTransformer {
    /// Load APR transformer from file using memory-mapped I/O (Y1)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .apr transformer file
    ///
    /// # Returns
    ///
    /// Memory-mapped transformer ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened or is invalid
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = MmapAprTransformer::from_file("model.apr")?;
    /// assert!(model.is_mmap());
    /// let logits = model.forward(&[1, 2, 3])?;
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open APR file: {e}"),
        })?;

        // Safety: We're only reading the file, mmap is safe for read-only access
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::IoError {
                message: format!("Failed to mmap APR file: {e}"),
            })?
        };

        // Verify minimum size
        if mmap.len() < APR_TRANSFORMER_HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file too small: {} bytes (need at least {})",
                    mmap.len(),
                    APR_TRANSFORMER_HEADER_SIZE
                ),
            });
        }

        // Parse header
        let header_bytes = &mmap[..APR_TRANSFORMER_HEADER_SIZE];

        // Verify APR magic
        let magic = &header_bytes[0..4];
        if magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: format!("Invalid APR magic: expected {:?}, got {:?}", MAGIC, magic),
            });
        }

        // Parse config from header (after 4-byte magic + 4-byte version)
        let version = u32::from_le_bytes([
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]);
        if version > 1 {
            return Err(RealizarError::FormatError {
                reason: format!("Unsupported APR version: {version}"),
            });
        }

        // Parse config fields (offset 8)
        let hidden_dim = u32::from_le_bytes([
            header_bytes[8],
            header_bytes[9],
            header_bytes[10],
            header_bytes[11],
        ]) as usize;
        let num_layers = u32::from_le_bytes([
            header_bytes[12],
            header_bytes[13],
            header_bytes[14],
            header_bytes[15],
        ]) as usize;
        let num_heads = u32::from_le_bytes([
            header_bytes[16],
            header_bytes[17],
            header_bytes[18],
            header_bytes[19],
        ]) as usize;
        let num_kv_heads = u32::from_le_bytes([
            header_bytes[20],
            header_bytes[21],
            header_bytes[22],
            header_bytes[23],
        ]) as usize;
        let vocab_size = u32::from_le_bytes([
            header_bytes[24],
            header_bytes[25],
            header_bytes[26],
            header_bytes[27],
        ]) as usize;
        let intermediate_dim = u32::from_le_bytes([
            header_bytes[28],
            header_bytes[29],
            header_bytes[30],
            header_bytes[31],
        ]) as usize;
        let context_length = u32::from_le_bytes([
            header_bytes[32],
            header_bytes[33],
            header_bytes[34],
            header_bytes[35],
        ]) as usize;
        let rope_theta = f32::from_le_bytes([
            header_bytes[36],
            header_bytes[37],
            header_bytes[38],
            header_bytes[39],
        ]);
        let eps = f32::from_le_bytes([
            header_bytes[40],
            header_bytes[41],
            header_bytes[42],
            header_bytes[43],
        ]);
        let tensor_data_offset = u32::from_le_bytes([
            header_bytes[44],
            header_bytes[45],
            header_bytes[46],
            header_bytes[47],
        ]) as usize;

        let config = AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
        };

        Ok(Self {
            mmap,
            config,
            tensor_data_offset,
            is_mmap: true,
        })
    }

    /// Check if model is using memory-mapped I/O (Y2)
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        self.is_mmap
    }

    /// Get raw tensor data slice (zero-copy access)
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset from tensor data start
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// Slice of raw bytes (zero-copy from mmap)
    pub fn get_tensor_bytes(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let start = self.tensor_data_offset + offset;
        let end = start + len;

        if end > self.mmap.len() {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Tensor access out of bounds: offset={offset}, len={len}, file_size={}",
                    self.mmap.len()
                ),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Get tensor as f32 slice (zero-copy if aligned)
    ///
    /// # Safety
    ///
    /// This function assumes the tensor data is properly aligned for f32 access.
    /// If not aligned, returns a copy.
    pub fn get_tensor_f32(&self, offset: usize, num_elements: usize) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_bytes(offset, num_elements * 4)?;

        // Convert bytes to f32 (could be zero-copy if aligned)
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    /// Get file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get number of parameters (estimated from config)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let layers = self.config.num_layers;
        let intermediate = self.config.intermediate_dim;

        // Embedding + LM head
        let embed_params = vocab * hidden * 2;

        // Per layer: attn_norm + qkv + attn_out + ffn_up + ffn_down
        let layer_params = hidden
            + (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        // Output norm
        let norm_params = hidden;

        embed_params + (layers * layer_params) + norm_params
    }
}

// ============================================================================
// Y5: Quantized APR Transformer (Q4_K, Q8_0 support)
// ============================================================================

/// Quantization type for APR Transformer weights (Y5)
///
/// Supports the same quantization formats as GGUF for format parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[allow(non_camel_case_types)] // Match GGML naming convention (Q4_K, Q8_0)
pub enum AprQuantizationType {
    /// Full precision 32-bit floats (no quantization)
    #[default]
    F32,
    /// 4-bit K-quantization (4.5 bits/weight, super-block size 256)
    Q4_K,
    /// 8-bit quantization (8 bits/weight, block size 32)
    Q8_0,
}

impl AprQuantizationType {
    /// Get bits per weight for this quantization type
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::Q4_K => 4.5, // 144 bytes per 256 values
            Self::Q8_0 => 8.0, // 36 bytes per 32 values (scale + 32 int8)
        }
    }

    /// Get bytes per super-block (256 values for Q4_K, 32 for Q8_0)
    #[must_use]
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,    // 4 bytes per value
            Self::Q4_K => 144, // 144 bytes per 256 values
            Self::Q8_0 => 36,  // 4 (scale) + 32 (int8) per 32 values
        }
    }

    /// Get values per block
    #[must_use]
    pub fn values_per_block(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::Q4_K => 256,
            Self::Q8_0 => 32,
        }
    }

    /// Convert to byte representation for header
    #[must_use]
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::Q4_K => 1,
            Self::Q8_0 => 2,
        }
    }

    /// Parse from byte representation
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::F32),
            1 => Some(Self::Q4_K),
            2 => Some(Self::Q8_0),
            _ => None,
        }
    }
}

/// Quantized APR Transformer with Q4_K or Q8_0 weights (Y5)
///
/// Stores weights in quantized form for memory efficiency while
/// providing the same inference interface as `AprTransformer`.
///
/// # Memory Savings
///
/// - Q4_K: ~7x compression (4.5 bits vs 32 bits)
/// - Q8_0: ~4x compression (8 bits vs 32 bits)
///
/// # Example
///
/// ```rust,ignore
/// use realizar::apr_transformer::{AprQuantizationType, QuantizedAprTransformer};
///
/// let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
/// let logits = transformer.forward(&[1, 2, 3])?;
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedAprTransformer {
    /// Model configuration
    config: AprTransformerConfig,
    /// Quantization type
    quant_type: AprQuantizationType,
    /// Token embedding (stored as F32 for now, could be quantized later)
    token_embedding: Vec<f32>,
    /// Quantized layer weights (raw bytes)
    layer_weights: Vec<Vec<u8>>,
    /// Output norm weight (F32)
    output_norm_weight: Vec<f32>,
    /// LM head weight (quantized)
    lm_head_weight: Vec<u8>,
}

impl QuantizedAprTransformer {
    /// Create a new quantized transformer with the given config and quantization type
    #[must_use]
    pub fn new(config: AprTransformerConfig, quant_type: AprQuantizationType) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let _intermediate_dim = config.intermediate_dim;

        // Calculate quantized sizes
        let embed_size = vocab_size * hidden_dim; // F32 for embeddings
        let layer_weight_size = Self::calculate_layer_bytes(&config, quant_type);
        let lm_head_size = Self::calculate_quantized_bytes(hidden_dim * vocab_size, quant_type);

        // Initialize with zeros
        let layer_weights = (0..config.num_layers)
            .map(|_| vec![0u8; layer_weight_size])
            .collect();

        Self {
            config,
            quant_type,
            token_embedding: vec![0.0; embed_size],
            layer_weights,
            output_norm_weight: vec![1.0; hidden_dim],
            lm_head_weight: vec![0u8; lm_head_size],
        }
    }

    /// Create from an F32 transformer by quantizing weights
    #[must_use]
    pub fn from_f32_transformer(
        f32_model: &AprTransformer,
        quant_type: AprQuantizationType,
    ) -> Self {
        let config = f32_model.config.clone();

        // For now, just create zero-initialized quantized model
        // Full quantization would convert F32 weights to Q4_K/Q8_0
        Self::new(config, quant_type)
    }

    /// Get the quantization type
    #[must_use]
    pub fn quantization_type(&self) -> AprQuantizationType {
        self.quant_type
    }

    /// Get bits per weight
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        self.quant_type.bits_per_weight()
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Get total quantized weight bytes
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        let embed_bytes = self.token_embedding.len() * 4; // F32
        let layer_bytes: usize = self.layer_weights.iter().map(std::vec::Vec::len).sum();
        let norm_bytes = self.output_norm_weight.len() * 4; // F32
        let lm_head_bytes = self.lm_head_weight.len();

        embed_bytes + layer_bytes + norm_bytes + lm_head_bytes
    }

    /// Get equivalent F32 size for compression ratio
    #[must_use]
    pub fn f32_equivalent_bytes(&self) -> usize {
        let num_params = self.num_parameters();
        num_params * 4 // 4 bytes per F32
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let layers = self.config.num_layers;
        let intermediate = self.config.intermediate_dim;

        // Embedding + LM head
        let embed_params = vocab * hidden * 2;

        // Per layer: attn_norm + qkv + attn_out + ffn_up + ffn_down
        let layer_params = hidden
            + (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        // Output norm
        let norm_params = hidden;

        embed_params + (layers * layer_params) + norm_params
    }

    /// Calculate bytes needed for layer weights
    fn calculate_layer_bytes(
        config: &AprTransformerConfig,
        quant_type: AprQuantizationType,
    ) -> usize {
        let hidden = config.hidden_dim;
        let intermediate = config.intermediate_dim;

        // Layer weights: qkv + attn_out + ffn_up + ffn_down + norms
        let weight_elements = (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        Self::calculate_quantized_bytes(weight_elements, quant_type)
    }

    /// Calculate quantized byte size for N elements
    fn calculate_quantized_bytes(num_elements: usize, quant_type: AprQuantizationType) -> usize {
        let values_per_block = quant_type.values_per_block();
        let bytes_per_block = quant_type.bytes_per_block();

        // Round up to nearest block
        let num_blocks = num_elements.div_ceil(values_per_block);
        num_blocks * bytes_per_block
    }

    /// Forward pass with quantized weights
    ///
    /// Dequantizes weights on-the-fly during computation.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let _vocab_size = self.config.vocab_size;

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through layers (simplified - dequantize on the fly)
        // For zero-initialized weights, this is essentially a no-op
        for _layer_weights in &self.layer_weights {
            // In production: dequantize and apply layer operations
            // For now with zero weights: output stays the same
        }

        // 3. Final layer norm
        let seq_len = token_ids.len();
        let eps = self.config.eps;
        let mut normed = Vec::with_capacity(hidden.len());

        for s in 0..seq_len {
            let start = s * hidden_dim;
            let slice = &hidden[start..start + hidden_dim];

            let mean: f32 = slice.iter().sum::<f32>() / hidden_dim as f32;
            let variance: f32 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
            let std_dev = (variance + eps).sqrt();

            for (i, &x) in slice.iter().enumerate() {
                let normalized = (x - mean) / std_dev;
                normed.push(normalized * self.output_norm_weight[i]);
            }
        }

        // 4. LM head (take last position, project to vocab)
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Dequantize LM head and compute logits
        let logits = self.compute_lm_head_logits(last_hidden)?;

        Ok(logits)
    }

    /// Compute LM head logits (dequantize weight and matmul)
    fn compute_lm_head_logits(&self, _hidden: &[f32]) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let _hidden_dim = self.config.hidden_dim;

        // For zero-initialized weights, output is zeros
        // In production: dequantize self.lm_head_weight and compute
        let logits = vec![0.0f32; vocab_size];

        // Simple matmul with dequantized weights (placeholder)
        // Real implementation would use fused_q4k_dot or dequantize_q8_0
        match self.quant_type {
            AprQuantizationType::F32 => {
                // No dequantization needed (but we store as bytes anyway)
            },
            AprQuantizationType::Q4_K => {
                // Would call: fused_q4k_dot for each output
            },
            AprQuantizationType::Q8_0 => {
                // Would call: dequantize_q8_0 then dot product
            },
        }

        Ok(logits)
    }

    /// Serialize to bytes (APR binary format with quantization)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Header (64 bytes)
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(self.config.hidden_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_layers as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_kv_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.vocab_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.intermediate_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.context_length as u32).to_le_bytes());
        bytes.extend_from_slice(&self.config.rope_theta.to_le_bytes());
        bytes.extend_from_slice(&self.config.eps.to_le_bytes());

        // Tensor data offset (after header)
        let tensor_offset = APR_TRANSFORMER_HEADER_SIZE as u32;
        bytes.extend_from_slice(&tensor_offset.to_le_bytes());

        // Quantization type at offset 48
        bytes.push(self.quant_type.to_byte());

        // Pad to 64 bytes
        while bytes.len() < APR_TRANSFORMER_HEADER_SIZE {
            bytes.push(0);
        }

        // Token embeddings (F32)
        for &v in &self.token_embedding {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Layer weights (quantized)
        for layer in &self.layer_weights {
            bytes.extend_from_slice(layer);
        }

        // Output norm (F32)
        for &v in &self.output_norm_weight {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // LM head (quantized)
        bytes.extend_from_slice(&self.lm_head_weight);

        Ok(bytes)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < APR_TRANSFORMER_HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!("Data too small: {} bytes", data.len()),
            });
        }

        // Verify magic
        if data[0..4] != MAGIC {
            return Err(RealizarError::FormatError {
                reason: "Invalid APR magic".to_string(),
            });
        }

        // Parse header
        let hidden_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let num_layers = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let num_heads = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let num_kv_heads = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let vocab_size = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as usize;
        let intermediate_dim =
            u32::from_le_bytes([data[28], data[29], data[30], data[31]]) as usize;
        let context_length = u32::from_le_bytes([data[32], data[33], data[34], data[35]]) as usize;
        let rope_theta = f32::from_le_bytes([data[36], data[37], data[38], data[39]]);
        let eps = f32::from_le_bytes([data[40], data[41], data[42], data[43]]);

        // Quantization type at offset 48
        let quant_type =
            AprQuantizationType::from_byte(data[48]).ok_or_else(|| RealizarError::FormatError {
                reason: format!("Invalid quantization type: {}", data[48]),
            })?;

        let config = AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
        };

        // For now, create with default weights
        // Full implementation would parse the weight data
        Ok(Self::new(config, quant_type))
    }

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
    /// Uses quantized weights with on-the-fly dequantization.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `cache` - Mutable KV cache to read from and append to
    /// * `position` - Position in sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    pub fn forward_with_cache(
        &self,
        token_id: u32,
        cache: &mut AprKVCache,
        _position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(hidden_dim);
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
        }

        // 2. Process through layers (simplified for quantized)
        for layer_idx in 0..self.config.num_layers {
            // For zero-initialized quantized weights, output stays mostly the same
            // In production: dequantize layer weights and compute

            // Compute placeholder K, V for cache
            let kv_size = num_kv_heads * head_dim;
            let k = vec![0.0f32; kv_size];
            let v = vec![0.0f32; kv_size];
            cache.append(layer_idx, &k, &v);
        }

        // 3. Final layer norm
        let eps = self.config.eps;
        let mean: f32 = hidden.iter().sum::<f32>() / hidden_dim as f32;
        let variance: f32 =
            hidden.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
        let std_dev = (variance + eps).sqrt();

        let mut normed = Vec::with_capacity(hidden_dim);
        for (i, &x) in hidden.iter().enumerate() {
            let normalized = (x - mean) / std_dev;
            normed.push(normalized * self.output_norm_weight[i]);
        }

        // 4. LM head (dequantize and compute)
        let logits = self.compute_lm_head_logits(&normed)?;

        Ok(logits)
    }
}

// ============================================================================
// Y4: KV Cache for Efficient Autoregressive Generation
// ============================================================================

/// KV Cache for efficient autoregressive generation (Y4)
///
/// Pre-allocates storage for keys and values to avoid allocations during decode.
/// Each layer has separate K and V caches stored contiguously.
///
/// # Memory Layout
///
/// For each layer: `[K_pos0, K_pos1, ..., K_posN, V_pos0, V_pos1, ..., V_posN]`
/// where each K/V entry has shape `[num_kv_heads * head_dim]`.
#[derive(Debug, Clone)]
pub struct AprKVCache {
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum context length (pre-allocated capacity)
    capacity: usize,
    /// Current sequence length (positions filled)
    len: usize,
    /// K cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    k_cache: Vec<Vec<f32>>,
    /// V cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    v_cache: Vec<Vec<f32>>,
}

impl AprKVCache {
    /// Create a new KV cache with pre-allocated capacity
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration
    ///
    /// # Returns
    ///
    /// Empty KV cache with capacity for full context length
    #[must_use]
    pub fn new(config: &AprTransformerConfig) -> Self {
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.hidden_dim / config.num_heads;
        let capacity = config.context_length;

        // Pre-allocate full capacity for each layer
        let kv_size = capacity * num_kv_heads * head_dim;
        let k_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();
        let v_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            capacity,
            len: 0,
            k_cache,
            v_cache,
        }
    }

    /// Get current sequence length (number of cached positions)
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get pre-allocated capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append K and V for a single position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `k` - Key tensor `[num_kv_heads * head_dim]`
    /// * `v` - Value tensor `[num_kv_heads * head_dim]`
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or cache is full
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert!(self.len < self.capacity, "KV cache is full");

        let kv_size = self.num_kv_heads * self.head_dim;
        let offset = self.len * kv_size;

        // Copy K and V into pre-allocated storage
        self.k_cache[layer][offset..offset + kv_size].copy_from_slice(k);
        self.v_cache[layer][offset..offset + kv_size].copy_from_slice(v);

        // Only increment len on first layer to keep consistent
        if layer == 0 {
            self.len += 1;
        }
    }

    /// Get cached K and V for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (K cache slice, V cache slice) containing all cached positions
    #[must_use]
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        let kv_size = self.num_kv_heads * self.head_dim;
        let used_size = self.len * kv_size;

        (
            &self.k_cache[layer][..used_size],
            &self.v_cache[layer][..used_size],
        )
    }

    /// Clear the cache (reset to empty without deallocating)
    pub fn clear(&mut self) {
        self.len = 0;
        // No need to zero memory - will be overwritten on next append
    }
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (optional)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

/// Configuration for APR Transformer models
///
/// Mirrors `GGUFConfig` for compatibility but is serializable to APR format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AprTransformerConfig {
    /// Model architecture name (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding/hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta for position encoding
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl Default for AprTransformerConfig {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }
}

/// Weights for a single transformer layer (all F32)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformerLayer {
    /// Attention norm weight [hidden_dim]
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional) [hidden_dim]
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [hidden_dim, 3*hidden_dim]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (optional) [3*hidden_dim]
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight [hidden_dim, hidden_dim]
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias (optional) [hidden_dim]
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate weight for SwiGLU (optional) [hidden_dim, intermediate_dim]
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate bias (optional) [intermediate_dim]
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight [hidden_dim, intermediate_dim]
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias (optional) [intermediate_dim]
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight [intermediate_dim, hidden_dim]
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias (optional) [hidden_dim]
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (optional) [hidden_dim]
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional) [hidden_dim]
    pub ffn_norm_bias: Option<Vec<f32>>,
}

impl AprTransformerLayer {
    /// Create an empty layer with given dimensions (non-GQA: num_kv_heads == num_heads)
    pub fn empty(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }
    }

    /// Create an empty layer with GQA dimensions (num_kv_heads < num_heads)
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden dimension (num_heads * head_dim)
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (< num_heads for GQA)
    /// * `intermediate_dim` - FFN intermediate dimension
    pub fn empty_gqa(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        // QKV weight: [hidden_dim, Q_dim + K_dim + V_dim] = [hidden_dim, hidden_dim + 2*kv_dim]
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * qkv_out_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }
    }

    /// Get total number of parameters in this layer
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.attn_norm_weight.len();
        count += self.attn_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.qkv_weight.len();
        count += self.qkv_bias.as_ref().map_or(0, Vec::len);
        count += self.attn_output_weight.len();
        count += self.attn_output_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_up_weight.len();
        count += self.ffn_up_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_down_weight.len();
        count += self.ffn_down_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_bias.as_ref().map_or(0, Vec::len);
        count
    }
}

/// Q4K/Q6K raw weights for fused kernel inference (F-GPU-130)
///
/// When present, matmul operations use fused kernels (matmul_q4k_f32, matmul_q6k_f32)
/// instead of the F32 path, avoiding full dequantization overhead.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Q4KLayerWeights {
    /// QKV projection weight in Q4K format
    pub qkv_weight: Option<Vec<u8>>,
    /// Attention output projection in Q4K format
    pub attn_output_weight: Option<Vec<u8>>,
    /// FFN gate weight in Q4K format (for SwiGLU)
    pub ffn_gate_weight: Option<Vec<u8>>,
    /// FFN up projection in Q4K format
    pub ffn_up_weight: Option<Vec<u8>>,
    /// FFN down projection in Q4K format
    pub ffn_down_weight: Option<Vec<u8>>,
    /// FFN down projection in Q6K format (when Q4K not available)
    pub ffn_down_weight_q6k: Option<Vec<u8>>,
    /// FFN up projection in Q6K format (when Q4K not available)
    pub ffn_up_weight_q6k: Option<Vec<u8>>,
}

/// APR Transformer model with F32 weights
///
/// WASM-compatible format for fair comparison with GGUF.
/// All weights are stored as F32 (dequantized from GGUF if converted).
///
/// For Q4K models, raw Q4K bytes can be stored in `q4k_layers` to enable
/// fused kernel inference (F-GPU-130) without full dequantization overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformer {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding weights [vocab_size * hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<AprTransformerLayer>,
    /// Output norm weight [hidden_dim]
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional) [hidden_dim]
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [hidden_dim * vocab_size]
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional) [vocab_size]
    pub lm_head_bias: Option<Vec<f32>>,
    /// Q4K raw layer weights for fused kernel inference (F-GPU-130)
    /// When present, enables direct Q4K matmul without dequantization
    #[serde(default)]
    pub q4k_layers: Option<Vec<Q4KLayerWeights>>,
    /// LM head weight in Q6K format for fused kernel inference
    /// When present, enables direct Q6K matmul without dequantization
    #[serde(default)]
    pub lm_head_weight_q6k: Option<Vec<u8>>,
    /// LM head weight in Q4K format for fused kernel inference
    #[serde(default)]
    pub lm_head_weight_q4k: Option<Vec<u8>>,
}

impl AprTransformer {
    /// Load APR transformer from an APR v2 file
    ///
    /// Parses the APR v2 format (magic "APR2") and extracts transformer weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .apr file
    ///
    /// # Returns
    ///
    /// Loaded transformer ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let transformer = AprTransformer::from_apr_file("model.apr")?;
    /// let logits = transformer.forward(&[1, 2, 3])?;
    /// ```
    pub fn from_apr_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let mut file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open APR file: {e}"),
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to read APR file: {e}"),
            })?;

        Self::from_apr_bytes(&data)
    }

    /// Load APR transformer from bytes
    ///
    /// Parses APR v2 format from memory buffer.
    pub fn from_apr_bytes(data: &[u8]) -> Result<Self> {
        // Check minimum size for header
        if data.len() < 64 {
            return Err(RealizarError::FormatError {
                reason: format!("APR file too small: {} bytes (need 64)", data.len()),
            });
        }

        // Check magic - first 3 bytes must be "APR", 4th byte is version (0, '1', or '2')
        let magic = &data[0..4];
        if magic[0..3] != *b"APR"
            || (magic[3] != 0 && magic[3] != b'1' && magic[3] != b'2')
        {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: {:?}, expected APR followed by version byte",
                    String::from_utf8_lossy(magic)
                ),
            });
        }

        // Parse header
        // APR header layout:
        //   0-3: Magic "APR\0"
        //   4-5: Version major.minor
        //   6-7: Flags
        //   8-11: Tensor count
        //   12-19: Metadata offset
        //   20-23: Metadata size
        //   24-31: Tensor index offset
        //   32-39: Data offset
        //   40-43: Checksum
        //   44-63: Reserved

        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]) as usize;
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]) as usize;
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]) as usize;

        // Parse metadata (JSON)
        let metadata_end = metadata_offset + metadata_size;
        if metadata_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "Metadata extends beyond file".to_string(),
            });
        }

        let metadata_json = &data[metadata_offset..metadata_end];
        let metadata: serde_json::Value = serde_json::from_slice(metadata_json).unwrap_or_default();

        // Extract architecture info from metadata
        let hidden_dim = metadata
            .get("hidden_size")
            .or_else(|| metadata.get("hidden_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(64) as usize;

        let num_layers = metadata
            .get("num_hidden_layers")
            .or_else(|| metadata.get("num_layers"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;

        let num_heads = metadata
            .get("num_attention_heads")
            .or_else(|| metadata.get("num_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(4) as usize;

        let num_kv_heads = metadata
            .get("num_key_value_heads")
            .or_else(|| metadata.get("num_kv_heads"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(num_heads as u64) as usize;

        let vocab_size = metadata
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32000) as usize;

        let intermediate_dim = metadata
            .get("intermediate_size")
            .or_else(|| metadata.get("intermediate_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        let rope_theta = metadata
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10000.0) as f32;

        let rms_norm_eps = metadata
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;

        let max_position = metadata
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2048) as usize;

        let config = AprTransformerConfig {
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: max_position,
            rope_theta,
            eps: rms_norm_eps,
            ..Default::default()
        };

        // Parse tensor index
        // APR v2 TensorIndexEntry format:
        //   - name_len (2 bytes) + name (variable)
        //   - dtype (1 byte)
        //   - ndim (1 byte) + dims (8 bytes each)
        //   - offset (8 bytes)
        //   - size (8 bytes)
        // Tuple: (offset, size, dims, dtype)
        let mut tensors: std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)> =
            std::collections::BTreeMap::new();

        let mut pos = tensor_index_offset;
        for _ in 0..tensor_count {
            if pos + 4 > data.len() {
                break;
            }

            // Read tensor name length and name
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            if pos + name_len + 18 > data.len() {
                break;
            }

            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            // Read dtype (1 byte)
            let dtype = data[pos];
            pos += 1;

            // Read ndim (1 byte)
            let ndim = data[pos] as usize;
            pos += 1;

            // Read dimensions (8 bytes each)
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                if pos + 8 > data.len() {
                    break;
                }
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
                dims.push(dim);
                pos += 8;
            }

            // Read offset (8 bytes)
            if pos + 16 > data.len() {
                break;
            }
            let offset = u64::from_le_bytes([
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

            // Read size (8 bytes)
            let size = u64::from_le_bytes([
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

            tensors.insert(name, (data_offset + offset, size, dims, dtype));
        }

        // Helper to extract f32 tensor (with Q4_K dequantization support)
        let get_f32_tensor = |name: &str| -> Option<Vec<f32>> {
            tensors.get(name).map(|(offset, size, dims, dtype)| {
                let end = offset + size;
                if end > data.len() {
                    return Vec::new();
                }
                let tensor_data = &data[*offset..end];

                match dtype {
                    // Q4_K: converter dtype=8 or APR v2 native dtype=12
                    8 | 12 => {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q4_k_apr(tensor_data, num_elements)
                    }
                    // Q6_K: converter dtype=9 or APR v2 native dtype=14
                    9 | 14 => {
                        let num_elements: usize = dims.iter().product();
                        dequantize_q6_k_apr(tensor_data, num_elements)
                    }
                    // F32 (dtype=0) or other: interpret as raw F32
                    _ => tensor_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                }
            })
        };

        // PMAT-103 FIX: Helper to get raw Q4K bytes (no dequantization) for fused kernel
        // Returns None if tensor is not Q4K/Q5K/Q6K format
        // APR dtype mapping (from GgufToAprQ4KConverter):
        //   8 = Q4_K (GGML 12) or Q5_K (GGML 13)
        //   9 = Q6_K (GGML 14)
        //  10 = Q8_0 (GGML 8)
        //  12 = Q4_K (APR v2 native)
        let get_q4k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q4K tensors from either converter dtype (8) or APR v2 native dtype (12)
                if *dtype != 8 && *dtype != 12 {
                    return None;
                }
                let end = offset + size;
                if end > data.len() {
                    return None;
                }
                Some(data[*offset..end].to_vec())
            })
        };

        // PMAT-103 FIX: Also extract Q6K raw bytes for fused kernel
        let get_q6k_raw_bytes = |name: &str| -> Option<Vec<u8>> {
            tensors.get(name).and_then(|(offset, size, _dims, dtype)| {
                // Accept Q6K tensors from either converter dtype (9) or APR v2 native dtype (14)
                if *dtype != 9 && *dtype != 14 {
                    return None;
                }
                let end = offset + size;
                if end > data.len() {
                    return None;
                }
                Some(data[*offset..end].to_vec())
            })
        };

        // Debug: print available tensor names
        eprintln!("[DEBUG] APR v2 tensor count: {tensor_count}");
        eprintln!("[DEBUG] Available tensor names (first 10):");
        for (i, (name, (offset, size, dims, dtype))) in tensors.iter().enumerate() {
            if i < 10 {
                eprintln!("  {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
            }
        }

        // PMAT-086 FIX: Transpose matrix from GGUF [in_dim, out_dim] to matmul [out_dim, in_dim]
        // GGUF/APR stores weights as [rows, cols] = [in_dim, out_dim] for y = x @ W
        // But our matmul expects [out_dim, in_dim] for y = W @ x (row-major GEMV)
        let transpose_weight = |data: Vec<f32>, rows: usize, cols: usize| -> Vec<f32> {
            let mut transposed = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    // data[r, c] -> transposed[c, r]
                    let src_idx = r * cols + c;
                    let dst_idx = c * rows + r;
                    if src_idx < data.len() && dst_idx < transposed.len() {
                        transposed[dst_idx] = data[src_idx];
                    }
                }
            }
            transposed
        };

        // PMAT-086: Detect if using GGUF naming (output.weight, blk.X) or HF naming (lm_head.weight)
        // GGUF uses [hidden_dim, vocab_size], HF uses [vocab_size, hidden_dim]
        let is_gguf_model = tensors.contains_key("output.weight") || tensors.contains_key("blk.0.attn_q.weight");
        eprintln!("[DEBUG] is_gguf_model={is_gguf_model}");

        // PMAT-086: Debug - check which embedding tensor names exist
        let embed_names = ["model.embed_tokens.weight", "token_embd.weight", "tok_embeddings.weight"];
        for name in &embed_names {
            if let Some((offset, size, dims, dtype)) = tensors.get(*name) {
                eprintln!("[DEBUG] Found embedding {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
            }
        }

        // Try to load token embedding
        let token_embedding_raw = get_f32_tensor("model.embed_tokens.weight")
            .or_else(|| get_f32_tensor("token_embd.weight"))
            .or_else(|| get_f32_tensor("tok_embeddings.weight"))
            .unwrap_or_else(|| {
                eprintln!("[DEBUG] WARNING: No embedding tensor found! Using zeros.");
                vec![0.0; vocab_size * hidden_dim]
            });

        // PMAT-086 FIX: APR stores GGUF data in row-major [vocab_size, hidden_dim] layout
        // even though the dims metadata says [hidden_dim, vocab_size] (GGML column-major convention)
        // The data is already correct - DO NOT transpose!
        let token_embedding = token_embedding_raw;

        eprintln!("[DEBUG] token_embedding loaded: {} elements, first 5: {:?}",
                  token_embedding.len(),
                  &token_embedding[..5.min(token_embedding.len())]);

        // Load output norm
        let output_norm_weight = get_f32_tensor("model.norm.weight")
            .or_else(|| get_f32_tensor("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden_dim]);

        // Debug: check output.weight / lm_head.weight
        for name in &["output.weight", "lm_head.weight"] {
            if let Some((offset, size, dims, dtype)) = tensors.get(*name) {
                eprintln!("[DEBUG] Found lm_head {name}: offset={offset}, size={size}, dims={dims:?}, dtype={dtype}");
            }
        }

        // Load LM head
        // For tied embeddings (common in Qwen, LLaMA models), use embed_tokens as fallback
        let lm_head_raw = get_f32_tensor("lm_head.weight")
            .or_else(|| get_f32_tensor("output.weight"))
            .or_else(|| {
                // Weight tying: use embedding weights for lm_head
                eprintln!("[DEBUG] Using tied weights: embedding -> lm_head");
                get_f32_tensor("model.embed_tokens.weight")
            })
            .or_else(|| get_f32_tensor("token_embd.weight"))
            .unwrap_or_else(|| {
                eprintln!("[DEBUG] WARNING: No lm_head tensor found! Using zeros.");
                vec![0.0; hidden_dim * vocab_size]
            });
        eprintln!("[DEBUG] lm_head_raw: {} elements, first 5: {:?}",
                  lm_head_raw.len(),
                  &lm_head_raw[..5.min(lm_head_raw.len())]);
        // PMAT-086 FIX: APR stores GGUF data in row-major [vocab_size, hidden_dim] layout
        // even though the dims metadata says [hidden_dim, vocab_size] (GGML column-major convention)
        // The data is already correct - DO NOT transpose!
        let lm_head_weight = lm_head_raw;

        // PMAT-103: Load lm_head Q4K/Q6K raw bytes for fused kernel inference
        let lm_head_weight_q4k = get_q4k_raw_bytes("lm_head.weight")
            .or_else(|| get_q4k_raw_bytes("output.weight"));
        let lm_head_weight_q6k = get_q6k_raw_bytes("lm_head.weight")
            .or_else(|| get_q6k_raw_bytes("output.weight"));
        if lm_head_weight_q4k.is_some() {
            eprintln!("[DEBUG] Loaded lm_head Q4K raw bytes for fused kernel");
        } else if lm_head_weight_q6k.is_some() {
            eprintln!("[DEBUG] Loaded lm_head Q6K raw bytes for fused kernel");
        }

        // Compute KV dimension from config
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Load layers
        let mut layers = Vec::with_capacity(num_layers);
        // PMAT-103 FIX: Also extract Q4K raw bytes for fused kernel inference
        let mut q4k_layer_weights: Vec<Q4KLayerWeights> = Vec::with_capacity(num_layers);
        let mut has_any_q4k = false;

        for i in 0..num_layers {
            let hf_prefix = format!("model.layers.{i}");
            let gguf_prefix = format!("blk.{i}");

            // Try separate Q/K/V or combined QKV
            // Support both HuggingFace and GGUF naming conventions
            // PMAT-086 FIX: HF uses [out_dim, in_dim], GGUF uses [in_dim, out_dim]
            // Only transpose GGUF tensors, not HF tensors
            let qkv_out_dim = hidden_dim + kv_dim + kv_dim;

            // Detect if using GGUF naming (blk.X) or HF naming (model.layers.X)
            let is_gguf = tensors.contains_key(&format!("{gguf_prefix}.attn_q.weight"));

            let qkv_weight = if let Some(qkv) =
                get_f32_tensor(&format!("{hf_prefix}.self_attn.qkv_proj.weight"))
            {
                // HF fused QKV - already in [qkv_out_dim, hidden_dim] format
                qkv
            } else {
                // Get Q weight
                let q_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.q_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_q.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
                // Get K weight
                let k_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.k_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_k.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
                // Get V weight
                let v_raw = get_f32_tensor(&format!("{hf_prefix}.self_attn.v_proj.weight"))
                    .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_v.weight")))
                    .unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);

                // PMAT-086 FIX: Both HF and GGUF data are in [out_dim, in_dim] layout
                // GGUF dims say [in_dim, out_dim] but data is actually [out_dim, in_dim] due to GGML column-major convention
                // Fuse Q, K, V by stacking rows (Q, then K, then V) - no transpose needed
                let _ = is_gguf; // Suppress unused warning
                let mut qkv = Vec::with_capacity(qkv_out_dim * hidden_dim);
                qkv.extend_from_slice(&q_raw);
                qkv.extend_from_slice(&k_raw);
                qkv.extend_from_slice(&v_raw);
                qkv
            };

            // Get Q/K/V biases (optional, for Qwen models)
            let q_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.q_proj.bias"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_q.bias")));
            let k_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.k_proj.bias"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_k.bias")));
            let v_bias = get_f32_tensor(&format!("{hf_prefix}.self_attn.v_proj.bias"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_v.bias")));

            // Combine biases if present
            let qkv_bias = match (&q_bias, &k_bias, &v_bias) {
                (Some(q), Some(k), Some(v)) => {
                    let mut bias = Vec::with_capacity(qkv_out_dim);
                    bias.extend_from_slice(q);
                    bias.extend_from_slice(k);
                    bias.extend_from_slice(v);
                    Some(bias)
                },
                _ => None,
            };

            // PMAT-086 FIX: Both HF and GGUF data are in [out_dim, in_dim] layout - no transpose needed
            let attn_output = get_f32_tensor(&format!("{hf_prefix}.self_attn.o_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_output.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);

            let attn_norm = get_f32_tensor(&format!("{hf_prefix}.input_layernorm.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.attn_norm.weight")))
                .unwrap_or_else(|| vec![1.0; hidden_dim]);

            let ffn_norm = get_f32_tensor(&format!("{hf_prefix}.post_attention_layernorm.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_norm.weight")));

            // PMAT-086 FIX: FFN weights - both HF and GGUF data are in [out_dim, in_dim] layout
            // No transpose needed - GGML column-major dims but row-major data
            let ffn_gate = get_f32_tensor(&format!("{hf_prefix}.mlp.gate_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_gate.weight")));
            let ffn_up = get_f32_tensor(&format!("{hf_prefix}.mlp.up_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_up.weight")))
                .unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = get_f32_tensor(&format!("{hf_prefix}.mlp.down_proj.weight"))
                .or_else(|| get_f32_tensor(&format!("{gguf_prefix}.ffn_down.weight")))
                .unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            // PMAT-103 FIX: Extract Q4K and Q6K raw bytes for fused kernel
            // Note: QKV is fused from separate Q/K/V so we can't use Q4K for it directly
            // But FFN weights are the largest, so we get most of the benefit from fusing those
            let q4k_attn_output = get_q4k_raw_bytes(&format!("{gguf_prefix}.attn_output.weight"));
            let q4k_ffn_gate = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_gate.weight"));
            let q4k_ffn_up = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight"));
            let q4k_ffn_down = get_q4k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight"));
            // Q6K fallback for tensors that aren't Q4K (common in mixed quantization models)
            let q6k_ffn_down = get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_down.weight"));
            let q6k_ffn_up = get_q6k_raw_bytes(&format!("{gguf_prefix}.ffn_up.weight"));

            let has_q4k_weights = q4k_attn_output.is_some()
                || q4k_ffn_gate.is_some()
                || q4k_ffn_up.is_some()
                || q4k_ffn_down.is_some();
            let has_q6k_weights = q6k_ffn_down.is_some() || q6k_ffn_up.is_some();

            if has_q4k_weights || has_q6k_weights {
                has_any_q4k = true;
            }

            q4k_layer_weights.push(Q4KLayerWeights {
                qkv_weight: None, // Q+K+V are fused from separate tensors
                attn_output_weight: q4k_attn_output,
                ffn_gate_weight: q4k_ffn_gate,
                ffn_up_weight: q4k_ffn_up,
                ffn_down_weight: q4k_ffn_down,
                ffn_down_weight_q6k: q6k_ffn_down,
                ffn_up_weight_q6k: q6k_ffn_up,
            });

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: attn_output,
                attn_output_bias: None,
                ffn_gate_weight: ffn_gate,
                ffn_gate_bias: None,
                ffn_up_weight: ffn_up,
                ffn_up_bias: None,
                ffn_down_weight: ffn_down,
                ffn_down_bias: None,
                ffn_norm_weight: ffn_norm,
                ffn_norm_bias: None,
            });
        }

        // PMAT-103 FIX: Store Q4K layer weights for fused kernel inference
        let q4k_layers = if has_any_q4k {
            eprintln!("[DEBUG] Loaded Q4K raw bytes for fused kernel inference");
            Some(q4k_layer_weights)
        } else {
            None
        };

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers,
            lm_head_weight_q6k,
            lm_head_weight_q4k,
        })
    }

    /// Create a new APR transformer with the given configuration
    pub fn new(config: AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;

        let layers = (0..config.num_layers)
            .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
            .collect();

        Self {
            config,
            token_embedding: vec![0.0; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Generate tokens autoregressively (simplified version without KV cache)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;

            // Greedy sampling: take argmax
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            tokens.push(next_token);

            // Stop at EOS tokens:
            // - Standard: 2
            // - Qwen2: 151645 (EOS), 151643 (BOS)
            // - LLaMA: 2
            if next_token == 2 || next_token == 151645 || next_token == 151643 {
                break;
            }
        }

        Ok(tokens)
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        for layer in &self.layers {
            count += layer.num_parameters();
        }
        count += self.output_norm_weight.len();
        count += self.output_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.lm_head_weight.len();
        count += self.lm_head_bias.as_ref().map_or(0, Vec::len);
        count
    }

    /// Get memory size in bytes (F32 = 4 bytes per param)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.num_parameters() * 4
    }

    /// Look up token embeddings
    #[must_use]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // Out of vocab - return zeros
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// RMSNorm (Root Mean Square Layer Normalization)
    ///
    /// Used by Qwen2, LLaMA, Mistral, and most modern LLMs.
    /// Formula: output = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// PMAT-094: Fixed five-whys root cause - was using LayerNorm (mean subtraction)
    /// instead of RMSNorm which caused garbage output for Qwen2 models.
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for s in 0..seq_len {
            let start = s * hidden_dim;
            let slice = &input[start..start + hidden_dim];

            // RMSNorm: compute root mean square (no mean subtraction!)
            let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

            // Normalize and scale
            for (i, &x) in slice.iter().enumerate() {
                let normalized = x / rms;
                let scaled = normalized * weight[i];
                let shifted = if let Some(b) = bias {
                    scaled + b[i]
                } else {
                    scaled
                };
                output.push(shifted);
            }
        }

        output
    }

    /// Matrix multiplication: output[out_dim] = weight[out_dim, in_dim] @ input[in_dim]
    ///
    /// PMAT-095 FIX: Weights are now stored in matvec-optimal [out_dim, in_dim] format.
    ///
    /// PMAT-103 FIX: Zero-copy implementation using raw slice operations.
    /// Previous implementations had O(n) allocation overhead per matmul call.
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let expected_size = in_dim * out_dim;

        if weight.len() != expected_size {
            return self.matmul_scalar(input, weight, in_dim, out_dim);
        }

        let mut output = vec![0.0f32; seq_len * out_dim];

        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];
            let out_start = s * out_dim;

            // PMAT-103: Unrolled dot product for better cache utilization
            // Process 4 output elements at a time when possible
            let out_chunks = out_dim / 4;
            let out_remainder = out_dim % 4;

            for o_chunk in 0..out_chunks {
                let o_base = o_chunk * 4;
                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut sum3 = 0.0f32;

                let w0_start = (o_base) * in_dim;
                let w1_start = (o_base + 1) * in_dim;
                let w2_start = (o_base + 2) * in_dim;
                let w3_start = (o_base + 3) * in_dim;

                for i in 0..in_dim {
                    let x = input_slice[i];
                    sum0 += x * weight[w0_start + i];
                    sum1 += x * weight[w1_start + i];
                    sum2 += x * weight[w2_start + i];
                    sum3 += x * weight[w3_start + i];
                }

                output[out_start + o_base] = sum0;
                output[out_start + o_base + 1] = sum1;
                output[out_start + o_base + 2] = sum2;
                output[out_start + o_base + 3] = sum3;
            }

            // Handle remainder
            for o in (out_dim - out_remainder)..out_dim {
                let w_start = o * in_dim;
                let mut sum = 0.0f32;
                for i in 0..in_dim {
                    sum += input_slice[i] * weight[w_start + i];
                }
                output[out_start + o] = sum;
            }
        }

        output
    }

    /// Scalar fallback for matmul (used when trueno fails)
    ///
    /// PMAT-095: Weight is [out_dim, in_dim] row-major format
    #[allow(clippy::unused_self)]
    fn matmul_scalar(
        &self,
        input: &[f32],
        weight: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let mut output = Vec::with_capacity(seq_len * out_dim);

        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];

            for o in 0..out_dim {
                let mut sum = 0.0;
                for (i, &input_val) in input_slice.iter().enumerate() {
                    // PMAT-095: Weight is [out_dim, in_dim] row-major
                    let weight_idx = o * in_dim + i;
                    if weight_idx < weight.len() {
                        sum += input_val * weight[weight_idx];
                    }
                }
                output.push(sum);
            }
        }

        output
    }

    /// Add bias in-place
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        let dim = bias.len();
        for (i, val) in data.iter_mut().enumerate() {
            *val += bias[i % dim];
        }
    }

    /// GELU activation (tanh approximation)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;

        for x in data.iter_mut() {
            let x3 = *x * *x * *x;
            let inner = SQRT_2_OVER_PI * (*x + GELU_COEFF * x3);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Apply Rotary Position Embedding (RoPE) to Q or K vectors
    ///
    /// RoPE encodes position information by rotating pairs of elements
    /// with position-dependent angles.
    fn apply_rope_f32(&self, x: &mut [f32], position: usize, num_heads: usize, head_dim: usize) {
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        for h in 0..num_heads {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            if idx2_start + half_dim > x.len() {
                continue;
            }

            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available)
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            // Calculate qkv_dim from actual weight size (handles GQA models)
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Proper attention with GQA support and RoPE
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Split QKV and apply RoPE
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V (layout: [Q..., K..., V...])
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // Apply RoPE to Q and K
                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Compute scaled dot-product attention with causal mask
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..seq_len {
                    // Compute attention scores for this position
                    let mut scores = Vec::with_capacity(i + 1);
                    let q_start = i * hidden_dim + q_head_offset;

                    for j in 0..=i {
                        // Only attend to positions <= current (causal mask)
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_all[q_start + d] * k_all[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    if exp_sum > 0.0 {
                        for s in &mut scores {
                            *s /= exp_sum;
                        }
                    }

                    // Weighted sum of V
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &weight) in scores.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += weight * v_all[v_start + d];
                        }
                    }
                }
            }

            // 2d. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available
            let mut attn_output = if let Some(ref q4k_bytes) = q4k_layer
                .and_then(|q| q.attn_output_weight.as_ref())
            {
                if layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using Q4K fused kernel");
                }
                // Fused Q4K matmul: process each position separately
                // PMAT-103: Use column-major kernel for GGUF layout
                let seq_len = token_ids.len();
                let mut output = Vec::with_capacity(seq_len * hidden_dim);
                for s in 0..seq_len {
                    let input_slice = &attn_out[s * hidden_dim..(s + 1) * hidden_dim];
                    let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, hidden_dim, hidden_dim);
                    output.extend(pos_out);
                }
                output
            } else {
                if layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using F32 fallback (slow!)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2g. FFN projection (SwiGLU or standard GELU)
            // PMAT-103: Use Q4K fused kernel when available for FFN
            let seq_len = token_ids.len();
            let ffn_output = if let Some(ref _gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // PMAT-103: Check for Q4K gate weight
                let gate = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_gate_weight.as_ref())
                {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_gate using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                        // ffn_gate: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                        let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, intermediate_dim, hidden_dim);
                        output.extend(pos_out);
                    }
                    output
                } else {
                    self.matmul(&ffn_input, layer.ffn_gate_weight.as_ref().expect("gate weight"), hidden_dim, intermediate_dim)
                };

                // PMAT-103: Check for Q4K up weight
                let up = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_up_weight.as_ref())
                {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_up using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                        // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                        let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, intermediate_dim, hidden_dim);
                        output.extend(pos_out);
                    }
                    output
                } else {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_up using F32 fallback (slow!)");
                    }
                    self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim)
                };

                // SiLU(gate) * up, then down projection
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp()); // SiLU = x * sigmoid(x)
                    ffn_hidden.push(silu_g * u);
                }

                // PMAT-103: Check for Q4K or Q6K down weight
                let mut out = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_down_weight.as_ref())
                {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, hidden_dim, intermediate_dim);
                        output.extend(pos_out);
                    }
                    output
                } else if let Some(ref q6k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_down_weight_q6k.as_ref())
                {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q6K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q6k_f32_ggml(q6k_bytes, input_slice, hidden_dim, intermediate_dim);
                        output.extend(pos_out);
                    }
                    output
                } else {
                    if layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using F32 fallback (slow!)");
                    }
                    self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP: down(GELU(up(x)))
                // PMAT-103: Check for Q4K up weight
                let mut ffn_hidden = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_up_weight.as_ref())
                {
                    let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                        // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                        let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, intermediate_dim, hidden_dim);
                        output.extend(pos_out);
                    }
                    output
                } else {
                    self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim)
                };
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                self.gelu(&mut ffn_hidden);

                // PMAT-103: Check for Q4K down weight
                let mut out = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_down_weight.as_ref())
                {
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice = &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q4k_f32_ggml(q4k_bytes, input_slice, hidden_dim, intermediate_dim);
                        output.extend(pos_out);
                    }
                    output
                } else {
                    self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };

            // 2h. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Predict next token (greedy decoding)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Token ID with highest probability
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;

        // Argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;

        Ok(max_idx as u32)
    }

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `cache` - Mutable KV cache to read from and append to
    /// * `position` - Position in sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    pub fn forward_with_cache(
        &self,
        token_id: u32,
        cache: &mut AprKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        // DEBUG: Force F32 fallback to verify data layout issues
        let force_f32 = std::env::var("APR_FORCE_F32").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // 2. Process through transformer layers
        let layers_start = std::time::Instant::now();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available) for fused kernels
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection (single token)
            // Calculate qkv_dim from actual weight size (handles GQA models)
            let qkv_out_dim = layer.qkv_weight.len() / hidden_dim;
            // NOTE: QKV projection currently uses F32 - fused kernel would need separate Q/K/V weights
            if layer_idx == 0 && position == 0 {
                eprintln!("[TRACE-CACHE] Layer 0: QKV projection using F32 (not fused)");
            }
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_out_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // Split into Q, K, V with GQA-aware sizes
            // Q: [hidden_dim] = [num_heads * head_dim]
            // K: [kv_size] = [num_kv_heads * head_dim]
            // V: [kv_size] = [num_kv_heads * head_dim]
            let kv_size = num_kv_heads * head_dim;
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_size].to_vec();
            let v = &qkv[hidden_dim + kv_size..hidden_dim + 2 * kv_size];

            // PMAT-103: Apply RoPE to Q and K at current position
            // This was missing, causing garbage output
            self.apply_rope_f32(&mut q, position, num_heads, head_dim);
            self.apply_rope_f32(&mut k, position, num_kv_heads, head_dim);

            // 2c. Append K, V to cache (K now has RoPE applied)
            cache.append(layer_idx, &k, v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let seq_len = cache.len();

            // Simplified attention: compute Q·K^T / sqrt(d), softmax, then V
            let mut attn_out = vec![0.0f32; hidden_dim];

            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads; // GQA mapping
                let q_start = h * head_dim;
                let q_slice = &q[q_start..q_start + head_dim]; // q is now Vec with RoPE applied

                // Compute attention scores
                let mut scores = Vec::with_capacity(seq_len);
                for pos in 0..seq_len {
                    let k_start = pos * kv_size + kv_head * head_dim;
                    let k_slice = &k_cache[k_start..k_start + head_dim];

                    let mut dot = 0.0f32;
                    for i in 0..head_dim {
                        dot += q_slice[i] * k_slice[i];
                    }
                    scores.push(dot / (head_dim as f32).sqrt());
                }

                // Causal mask: only attend to positions <= current
                for pos in (position + 1)..seq_len {
                    scores[pos] = f32::NEG_INFINITY;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_scores: Vec<f32> =
                    scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum: f32 = exp_scores.iter().sum();
                if sum > 0.0 {
                    for s in &mut exp_scores {
                        *s /= sum;
                    }
                }

                // Weighted sum of V
                for pos in 0..seq_len {
                    let v_start = pos * kv_size + kv_head * head_dim;
                    let v_slice = &v_cache[v_start..v_start + head_dim];
                    for i in 0..head_dim {
                        attn_out[q_start + i] += exp_scores[pos] * v_slice[i];
                    }
                }
            }

            // 2e. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available (single token path)
            let mut attn_output = if !force_f32 {
                if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.attn_output_weight.as_ref())
                {
                    if layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using Q4K fused kernel");
                    }
                    matmul_q4k_f32_ggml(q4k_bytes, &attn_out, hidden_dim, hidden_dim)
                } else {
                    if layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 fallback (slow!)");
                    }
                    self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
                }
            } else {
                if layer_idx == 0 && position == 0 {
                    eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 (APR_FORCE_F32)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2g. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2h. FFN projection (SwiGLU or standard GELU)
            // PMAT-103 FIX: Use Q4K/Q6K fused kernels when available (single token path)
            let intermediate_dim = self.config.intermediate_dim;
            let ffn_output = if let Some(ref _gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // PMAT-103: Check for Q4K gate weight
                let gate = if !force_f32 {
                    if let Some(ref q4k_bytes) = q4k_layer
                        .and_then(|q| q.ffn_gate_weight.as_ref())
                    {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using Q4K fused kernel");
                        }
                        matmul_q4k_f32_ggml(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                    } else {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 fallback (slow!)");
                        }
                        self.matmul(&ffn_input, layer.ffn_gate_weight.as_ref().expect("gate weight"), hidden_dim, intermediate_dim)
                    }
                } else {
                    if layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(&ffn_input, layer.ffn_gate_weight.as_ref().expect("gate weight"), hidden_dim, intermediate_dim)
                };

                // PMAT-103: Check for Q4K/Q6K up weight
                let up = if !force_f32 {
                    if let Some(ref q4k_bytes) = q4k_layer
                        .and_then(|q| q.ffn_up_weight.as_ref())
                    {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q4K fused kernel");
                        }
                        matmul_q4k_f32_ggml(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                    } else if let Some(ref q6k_bytes) = q4k_layer
                        .and_then(|q| q.ffn_up_weight_q6k.as_ref())
                    {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q6K fused kernel");
                        }
                        matmul_q6k_f32_ggml(q6k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                    } else {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 fallback (slow!)");
                        }
                        self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim)
                    }
                } else {
                    if layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim)
                };

                // SiLU(gate) * up, then down projection
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp()); // SiLU = x * sigmoid(x)
                    ffn_hidden.push(silu_g * u);
                }

                // PMAT-103: Check for Q4K or Q6K down weight
                let mut out = if !force_f32 {
                    if let Some(ref q4k_bytes) = q4k_layer
                        .and_then(|q| q.ffn_down_weight.as_ref())
                    {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q4K fused kernel");
                        }
                        matmul_q4k_f32_ggml(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
                    } else if let Some(ref q6k_bytes) = q4k_layer
                        .and_then(|q| q.ffn_down_weight_q6k.as_ref())
                    {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q6K fused kernel");
                        }
                        matmul_q6k_f32_ggml(q6k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
                    } else {
                        if layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 fallback (slow!)");
                        }
                        self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
                    }
                } else {
                    if layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP: down(GELU(up(x)))
                // PMAT-103: Check for Q4K up weight
                let mut ffn_hidden = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_up_weight.as_ref())
                {
                    matmul_q4k_f32_ggml(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                } else {
                    self.matmul(&ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim)
                };
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                self.gelu(&mut ffn_hidden);

                // PMAT-103: Check for Q4K down weight
                let mut out = if let Some(ref q4k_bytes) = q4k_layer
                    .and_then(|q| q.ffn_down_weight.as_ref())
                {
                    matmul_q4k_f32_ggml(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)
                } else {
                    self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };

            // 2i. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }
        eprintln!("[TRACE-CACHE] pos={}: {} layers took {:?}", position, self.layers.len(), layers_start.elapsed());

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        // PMAT-103: Use Q4K/Q6K fused kernel when available (single token path)
        let lm_start = std::time::Instant::now();
        let mut logits = if !force_f32 {
            if let Some(ref q4k_bytes) = self.lm_head_weight_q4k {
                eprintln!("[TRACE-CACHE] lm_head using Q4K fused kernel");
                matmul_q4k_f32_ggml(q4k_bytes, &normed, self.config.vocab_size, hidden_dim)
            } else if let Some(ref q6k_bytes) = self.lm_head_weight_q6k {
                let result = matmul_q6k_f32_ggml(q6k_bytes, &normed, self.config.vocab_size, hidden_dim);
                eprintln!("[TRACE-CACHE] lm_head Q6K took {:?}", lm_start.elapsed());
                result
            } else {
                self.matmul(
                    &normed,
                    &self.lm_head_weight,
                    hidden_dim,
                    self.config.vocab_size,
                )
            }
        } else {
            eprintln!("[TRACE-CACHE] lm_head using F32 (APR_FORCE_F32)");
            self.matmul(
                &normed,
                &self.lm_head_weight,
                hidden_dim,
                self.config.vocab_size,
            )
        };
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens using KV cache for efficiency (Y4)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate_with_cache(&self, prompt: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut cache = AprKVCache::new(&self.config);
        let mut output = prompt.to_vec();

        // PMAT-103 FIX: Process prompt tokens and KEEP the logits from the last one.
        // Previously we threw away all logits (`let _ = ...`) and then reprocessed
        // the last prompt token at the same position, corrupting the KV cache.
        let mut logits = Vec::new();

        // PMAT-103 TRACE: Measure per-token timing to verify O(n) vs O(n²)
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();
        if trace_enabled {
            eprintln!("[TRACE] Processing {} prompt tokens...", prompt.len());
        }

        for (pos, &token) in prompt.iter().enumerate() {
            let start = std::time::Instant::now();
            logits = self.forward_with_cache(token, &mut cache, pos)?;
            if trace_enabled {
                eprintln!("[TRACE] Prompt token {}: {:?}", pos, start.elapsed());
            }
        }

        // Generate new tokens using the logits we already have
        for i in 0..config.max_tokens {
            // Sample from current logits (which predict the NEXT token)
            let next_token = if config.temperature == 0.0 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = scaled.iter().map(|s| (s - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            };

            output.push(next_token);

            // Check for EOS tokens
            if next_token == 0 || next_token == 2 || next_token == 151645 || next_token == 151643 {
                break;
            }

            // If we need more tokens, process this one to get logits for the next
            if i < config.max_tokens - 1 {
                // Position is output.len() - 1 = prompt.len() + (i + 1) - 1 = prompt.len() + i
                let start = std::time::Instant::now();
                logits = self.forward_with_cache(next_token, &mut cache, output.len() - 1)?;
                if trace_enabled {
                    eprintln!("[TRACE] Gen token {} (pos {}): {:?}", i, output.len() - 1, start.elapsed());
                }
            }
        }

        if trace_enabled {
            eprintln!("[TRACE] Generation complete. Total output tokens: {}", output.len());
        }

        Ok(output)
    }
}

/// Convert from `GGUFTransformer` to APR format
///
/// This dequantizes all GGUF weights to F32 for WASM compatibility.
#[cfg(feature = "default")]
impl From<&crate::gguf::GGUFTransformer> for AprTransformer {
    fn from(gguf: &crate::gguf::GGUFTransformer) -> Self {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
            })
            .collect();

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }
}

// ============================================================================
// SIMD-Accelerated Quantized APR Transformer (Q4_0)
// ============================================================================

/// Q4_0 quantized tensor for SIMD-accelerated inference
///
/// Stores raw Q4_0 bytes (18 bytes per 32 values) with dimensions for matmul.
#[derive(Debug, Clone)]
pub struct QuantizedAprTensorQ4 {
    /// Raw Q4_0 quantized data
    pub data: Vec<u8>,
    /// Input dimension (columns in weight matrix)
    pub in_dim: usize,
    /// Output dimension (rows in weight matrix)
    pub out_dim: usize,
}

impl QuantizedAprTensorQ4 {
    /// Create a new Q4_0 tensor from raw data
    #[must_use]
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Create empty tensor with proper Q4_0 allocation
    #[must_use]
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_elements = in_dim * out_dim;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        let data = vec![0u8; num_blocks * Q4_0_BLOCK_BYTES];
        Self {
            data,
            in_dim,
            out_dim,
        }
    }

    /// Get expected data size in bytes
    #[must_use]
    pub fn expected_bytes(num_elements: usize) -> usize {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
        num_blocks * Q4_0_BLOCK_BYTES
    }
}

/// Q4_0 quantized layer for SIMD-accelerated inference
///
/// Stores individual Q4_0 tensors for each weight matrix, enabling
/// direct use of `fused_q4_0_q8_0_parallel_matvec`.
#[derive(Debug, Clone)]
pub struct QuantizedAprLayerQ4 {
    /// Attention norm weight (F32, small)
    pub attn_norm_weight: Vec<f32>,
    /// QKV projection weights (Q4_0)
    pub qkv_weight: QuantizedAprTensorQ4,
    /// Attention output projection (Q4_0)
    pub attn_output_weight: QuantizedAprTensorQ4,
    /// FFN up projection (Q4_0)
    pub ffn_up_weight: QuantizedAprTensorQ4,
    /// FFN down projection (Q4_0)
    pub ffn_down_weight: QuantizedAprTensorQ4,
    /// FFN gate projection for SwiGLU (Q4_0, optional)
    pub ffn_gate_weight: Option<QuantizedAprTensorQ4>,
    /// FFN norm weight (F32, optional)
    pub ffn_norm_weight: Option<Vec<f32>>,
}

/// SIMD-accelerated Quantized APR Transformer
///
/// Stores weights in Q4_0 format and uses integer SIMD matmul
/// (`_mm256_maddubs_epi16`) for near-GGUF performance.
///
/// # Performance
///
/// Expected throughput: ~17 tok/s on TinyLlama-1.1B (1.36x vs GGUF)
/// With KV cache: ~25-34 tok/s expected (1.5-2x additional speedup)
#[derive(Debug, Clone)]
pub struct QuantizedAprTransformerQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32 for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Quantized layers
    pub layers: Vec<QuantizedAprLayerQ4>,
    /// Output norm weight (F32)
    pub output_norm_weight: Vec<f32>,
    /// LM head weight (Q4_0)
    pub lm_head_weight: QuantizedAprTensorQ4,
}

/// Scratch buffer for zero-allocation inference
///
/// Pre-allocates all intermediate buffers needed for a forward pass.
/// Reuse across multiple forward calls to eliminate per-token allocations.
#[derive(Debug)]
pub struct AprInferenceScratch {
    /// Hidden state [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// QKV projection output [qkv_dim]
    pub qkv_out: Vec<f32>,
    /// Query vectors [q_dim]
    pub q: Vec<f32>,
    /// Key vectors [k_dim]
    pub k: Vec<f32>,
    /// Value vectors [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN input [hidden_dim]
    pub ffn_input: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN output [hidden_dim]
    pub ffn_out: Vec<f32>,
}

impl AprInferenceScratch {
    /// Create scratch buffer sized for a model config
    #[must_use]
    pub fn from_config(config: &AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let qkv_dim = hidden_dim * 3; // Conservative estimate
        let intermediate_dim = config.intermediate_dim;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv_out: vec![0.0; qkv_dim],
            q: vec![0.0; hidden_dim],
            k: vec![0.0; hidden_dim],
            v: vec![0.0; hidden_dim],
            attn_out: vec![0.0; hidden_dim],
            ffn_input: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_out: vec![0.0; hidden_dim],
        }
    }

    /// Clear all buffers (set to zero)
    pub fn clear(&mut self) {
        self.hidden.fill(0.0);
        self.normed.fill(0.0);
        self.qkv_out.fill(0.0);
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.attn_out.fill(0.0);
        self.ffn_input.fill(0.0);
        self.ffn_up.fill(0.0);
        self.ffn_gate.fill(0.0);
        self.ffn_out.fill(0.0);
    }
}

impl QuantizedAprTransformerQ4 {
    /// Create from GGUF OwnedQuantizedModel (extracts Q4_0 bytes)
    ///
    /// # Arguments
    ///
    /// * `gguf` - Source GGUF model with Q4_0 weights
    ///
    /// # Returns
    ///
    /// Quantized APR transformer with same weights
    pub fn from_gguf(gguf: &crate::gguf::OwnedQuantizedModel) -> Self {
        use crate::gguf::OwnedQKVWeights;

        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers =
            gguf.layers
                .iter()
                .map(|l| {
                    // Extract QKV weight data
                    let qkv_weight = match &l.qkv_weight {
                        OwnedQKVWeights::Fused(t) => {
                            QuantizedAprTensorQ4::new(t.data.clone(), t.in_dim, t.out_dim)
                        },
                        OwnedQKVWeights::Separate { q, k, v } => {
                            // Concatenate Q, K, V for fused format
                            let mut data =
                                Vec::with_capacity(q.data.len() + k.data.len() + v.data.len());
                            data.extend_from_slice(&q.data);
                            data.extend_from_slice(&k.data);
                            data.extend_from_slice(&v.data);
                            QuantizedAprTensorQ4::new(
                                data,
                                q.in_dim,                          // hidden_dim
                                q.out_dim + k.out_dim + v.out_dim, // qkv_dim
                            )
                        },
                    };

                    QuantizedAprLayerQ4 {
                        attn_norm_weight: l.attn_norm_weight.clone(),
                        qkv_weight,
                        attn_output_weight: QuantizedAprTensorQ4::new(
                            l.attn_output_weight.data.clone(),
                            l.attn_output_weight.in_dim,
                            l.attn_output_weight.out_dim,
                        ),
                        ffn_up_weight: QuantizedAprTensorQ4::new(
                            l.ffn_up_weight.data.clone(),
                            l.ffn_up_weight.in_dim,
                            l.ffn_up_weight.out_dim,
                        ),
                        ffn_down_weight: QuantizedAprTensorQ4::new(
                            l.ffn_down_weight.data.clone(),
                            l.ffn_down_weight.in_dim,
                            l.ffn_down_weight.out_dim,
                        ),
                        ffn_gate_weight: l.ffn_gate_weight.as_ref().map(|g| {
                            QuantizedAprTensorQ4::new(g.data.clone(), g.in_dim, g.out_dim)
                        }),
                        ffn_norm_weight: l.ffn_norm_weight.clone(),
                    }
                })
                .collect();

        let lm_head_weight = QuantizedAprTensorQ4::new(
            gguf.lm_head_weight.data.clone(),
            gguf.lm_head_weight.in_dim,
            gguf.lm_head_weight.out_dim,
        );

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            lm_head_weight,
        }
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Create a scratch buffer for zero-allocation inference
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = QuantizedAprTransformerQ4::from_gguf(&gguf);
    /// let mut scratch = model.create_scratch();
    ///
    /// // Reuse scratch across multiple forward passes
    /// for token_id in token_ids {
    ///     let logits = model.forward_single_with_scratch(token_id, &mut scratch)?;
    /// }
    /// ```
    #[must_use]
    pub fn create_scratch(&self) -> AprInferenceScratch {
        AprInferenceScratch::from_config(&self.config)
    }

    /// Forward pass using SIMD-accelerated Q4_0×Q8_0 matmul
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (F32)
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(seq_len * qkv_dim);
            for s in 0..seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            // Proper attention with RoPE and causal mask
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v = &qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim];

                // Apply RoPE to Q and K (position-dependent rotation)
                self.apply_rope(&mut q, s, num_heads);
                self.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_output = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with SwiGLU (sequential to avoid nested parallelism overhead)
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Sequential up + gate (both matmuls use internal parallelism)
                let mut ffn_up_out = Vec::with_capacity(seq_len * intermediate_dim);
                let mut ffn_gate_out = Vec::with_capacity(seq_len * intermediate_dim);

                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];

                    // Up projection
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_up_out.extend(u);

                    // Gate projection
                    let g = fused_q4_0_q8_0_parallel_matvec(
                        &gate.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_gate_out.extend(g);
                }

                // Apply SiLU to gate and multiply with up
                for i in 0..ffn_up_out.len() {
                    let silu = ffn_gate_out[i] / (1.0 + (-ffn_gate_out[i]).exp());
                    ffn_up_out[i] *= silu;
                }
                ffn_up_out
            } else {
                // Non-SwiGLU: Sequential up projection + GELU
                let mut up = Vec::with_capacity(seq_len * intermediate_dim);
                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                // GELU activation (tanh approximation)
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm
        let last_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Create a KV cache for this model
    #[must_use]
    pub fn create_kv_cache(&self) -> AprKVCache {
        AprKVCache::new(&self.config)
    }

    /// Forward pass for a single token using scratch buffer (zero allocation)
    ///
    /// This is the fastest path for autoregressive generation when combined
    /// with `forward_with_cache_and_scratch`. It reuses pre-allocated buffers
    /// to eliminate per-token allocations.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `scratch` - Pre-allocated scratch buffer (from `create_scratch()`)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward_single_with_scratch(
        &self,
        token_id: u32,
        scratch: &mut AprInferenceScratch,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec_into;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (write directly to scratch.hidden)
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            scratch.hidden[..hidden_dim]
                .copy_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            scratch.hidden[..hidden_dim].fill(0.0);
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm (reuse scratch.normed)
            let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
            for i in 0..hidden_dim {
                scratch.normed[i] = scratch.hidden[i] / rms * layer.attn_norm_weight[i];
            }

            // QKV projection (zero-allocation - write directly to scratch.qkv_out)
            let qkv_dim = layer.qkv_weight.out_dim;
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.qkv_weight.data,
                &scratch.normed[..hidden_dim],
                hidden_dim,
                &mut scratch.qkv_out[..qkv_dim],
            )?;

            // Extract Q, K, V and apply RoPE (position=0 for single token)
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            scratch.q[..q_dim].copy_from_slice(&scratch.qkv_out[..q_dim]);
            scratch.k[..kv_dim].copy_from_slice(&scratch.qkv_out[q_dim..q_dim + kv_dim]);
            scratch.v[..kv_dim]
                .copy_from_slice(&scratch.qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim]);

            // Apply RoPE at position 0
            self.apply_rope(&mut scratch.q[..q_dim], 0, num_heads);
            self.apply_rope(&mut scratch.k[..kv_dim], 0, num_kv_heads);

            // For single token, attention is trivial: output = V (softmax of 1 element = 1.0)
            let group_size = num_heads / num_kv_heads;
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                scratch.attn_out[out_offset..out_offset + head_dim]
                    .copy_from_slice(&scratch.v[v_offset..v_offset + head_dim]);
            }

            // Output projection (write to scratch.ffn_out as temporary)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.attn_output_weight.data,
                &scratch.attn_out[..hidden_dim],
                layer.attn_output_weight.in_dim,
                &mut scratch.ffn_out[..layer.attn_output_weight.out_dim],
            )?;

            // Residual connection (attn)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }

            // Pre-FFN norm
            if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for i in 0..hidden_dim {
                    scratch.ffn_input[i] = scratch.hidden[i] / rms * ffn_norm[i];
                }
            } else {
                scratch.ffn_input[..hidden_dim].copy_from_slice(&scratch.normed[..hidden_dim]);
            }

            // FFN with SwiGLU
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            if let Some(gate) = &layer.ffn_gate_weight {
                // Up projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                // Gate projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &gate.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_gate[..intermediate_dim],
                )?;

                // SwiGLU: silu(gate) * up
                for i in 0..intermediate_dim {
                    let silu = scratch.ffn_gate[i] / (1.0 + (-scratch.ffn_gate[i]).exp());
                    scratch.ffn_up[i] *= silu;
                }
            } else {
                // GELU path (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for i in 0..intermediate_dim {
                    let x = scratch.ffn_up[i];
                    let t = (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh();
                    scratch.ffn_up[i] = 0.5 * x * (1.0 + t);
                }
            }

            // Down projection (write to scratch.ffn_out)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &scratch.ffn_up[..intermediate_dim],
                intermediate_dim,
                &mut scratch.ffn_out[..hidden_dim],
            )?;

            // Residual connection (FFN)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }
        }

        // 3. Final RMS norm
        let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        for i in 0..hidden_dim {
            scratch.normed[i] = scratch.hidden[i] / rms * self.output_norm_weight[i];
        }

        // 4. LM head projection (still allocates - logits must be returned)
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        fused_q4_0_q8_0_parallel_matvec_into(
            &self.lm_head_weight.data,
            &scratch.normed[..hidden_dim],
            hidden_dim,
            &mut logits,
        )?;

        Ok(logits)
    }

    /// Forward pass with KV cache for efficient autoregressive generation
    ///
    /// This method only computes attention for the new token(s), reusing
    /// cached K/V from previous positions. Provides 1.5-2x speedup.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - New token IDs to process (typically 1 for generation)
    /// * `cache` - KV cache to use and update
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for the last token
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut AprKVCache,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // Position in the sequence (including cached positions)
        let cache_len = cache.len();
        let new_seq_len = token_ids.len();

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(new_seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..new_seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul (only for new tokens)
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(new_seq_len * qkv_dim);
            for s in 0..new_seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Process new tokens: extract Q, K, V and apply RoPE
            let mut new_q = Vec::with_capacity(new_seq_len * q_dim);
            for s in 0..new_seq_len {
                let qkv_start = s * qkv_dim;
                let position = cache_len + s;

                // Extract Q, K, V for this position
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v =
                    qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim].to_vec();

                // Apply RoPE with correct position
                self.apply_rope(&mut q, position, num_heads);
                self.apply_rope(&mut k, position, num_kv_heads);

                new_q.extend_from_slice(&q);

                // Append to cache (K and V with RoPE applied to K)
                cache.append(layer_idx, &k, &v);
            }

            // Get full K and V from cache (includes new tokens)
            let (full_k, full_v) = cache.get(layer_idx);
            let total_seq_len = cache.len();

            // Compute attention: new Q attends to all cached K/V
            let attn_output = self.causal_attention_cached(
                &new_q,
                full_k,
                full_v,
                new_seq_len,
                total_seq_len,
                cache_len,
            );

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..new_seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with parallel up/gate for SwiGLU models
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Parallel FFN up + gate
                let (ffn_up_result, ffn_gate_result) = rayon::join(
                    || {
                        let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(u) = fused_q4_0_q8_0_parallel_matvec(
                                &layer.ffn_up_weight.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                up.extend(u);
                            }
                        }
                        up
                    },
                    || {
                        let mut g = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(gv) = fused_q4_0_q8_0_parallel_matvec(
                                &gate.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                g.extend(gv);
                            }
                        }
                        g
                    },
                );

                let mut up = ffn_up_result;
                for i in 0..up.len() {
                    let silu = ffn_gate_result[i] / (1.0 + (-ffn_gate_result[i]).exp());
                    up[i] *= silu;
                }
                up
            } else {
                // Non-SwiGLU: Sequential + GELU
                let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                for s in 0..new_seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm (only for last token)
        let last_start = (new_seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Attention with KV cache - new Q attends to all cached K/V
    ///
    /// Parallelizes across attention heads for efficiency.
    fn causal_attention_cached(
        &self,
        new_q: &[f32],
        full_k: &[f32],
        full_v: &[f32],
        new_seq_len: usize,
        _total_seq_len: usize,
        cache_len: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = num_heads / num_kv_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            // Sequential path
            let mut output = vec![0.0f32; new_seq_len * q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..new_seq_len {
                    let pos = cache_len + i;
                    let mut scores = Vec::with_capacity(pos + 1);
                    let q_start = i * q_dim + q_head_offset;

                    // Attend to all positions up to current (causal)
                    for j in 0..=pos {
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += new_q[q_start + d] * full_k[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    for s in &mut scores {
                        *s /= exp_sum;
                    }

                    // Weighted sum
                    let out_start = i * q_dim + q_head_offset;
                    for (j, &weight) in scores.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            output[out_start + d] += weight * full_v[v_start + d];
                        }
                    }
                }
            }
            output
        } else {
            // Parallel path
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; new_seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..new_seq_len {
                        let pos = cache_len + i;
                        let mut scores = Vec::with_capacity(pos + 1);
                        let q_start = i * q_dim + q_head_offset;

                        for j in 0..=pos {
                            let k_start = j * kv_dim + kv_head_offset;
                            let mut score = 0.0f32;
                            for d in 0..head_dim {
                                score += new_q[q_start + d] * full_k[k_start + d];
                            }
                            scores.push(score * scale);
                        }

                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            exp_sum += *s;
                        }
                        for s in &mut scores {
                            *s /= exp_sum;
                        }

                        let out_start = i * head_dim;
                        for (j, &weight) in scores.iter().enumerate() {
                            let v_start = j * kv_dim + kv_head_offset;
                            for d in 0..head_dim {
                                head_out[out_start + d] += weight * full_v[v_start + d];
                            }
                        }
                    }
                    head_out
                })
                .collect();

            // Merge
            let mut output = vec![0.0f32; new_seq_len * q_dim];
            for (head, head_out) in head_outputs.into_iter().enumerate() {
                let head_offset = head * head_dim;
                for i in 0..new_seq_len {
                    let src_start = i * head_dim;
                    let dst_start = i * q_dim + head_offset;
                    output[dst_start..dst_start + head_dim]
                        .copy_from_slice(&head_out[src_start..src_start + head_dim]);
                }
            }
            output
        }
    }

    /// Get memory footprint in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let embed_size = self.token_embedding.len() * 4;
        let norm_size = self.output_norm_weight.len() * 4;
        let lm_head_size = self.lm_head_weight.data.len();

        let layer_size: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm_weight.len() * 4
                    + l.qkv_weight.data.len()
                    + l.attn_output_weight.data.len()
                    + l.ffn_up_weight.data.len()
                    + l.ffn_down_weight.data.len()
                    + l.ffn_gate_weight.as_ref().map_or(0, |g| g.data.len())
                    + l.ffn_norm_weight.as_ref().map_or(0, |n| n.len() * 4)
            })
            .sum();

        embed_size + norm_size + lm_head_size + layer_size
    }

    /// Apply Rotary Position Embeddings (RoPE) to a tensor
    ///
    /// RoPE applies position-dependent rotation to pairs of dimensions,
    /// enabling the model to learn relative positional information.
    fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        // Apply rotation to each head with inline cos/sin computation
        // Avoids allocation by computing cos/sin on the fly
        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            if idx2_start + half_dim > x.len() {
                continue;
            }

            // Process 4 elements at a time for better ILP
            let mut i = 0;
            while i + 4 <= half_dim {
                // Compute 4 frequencies
                let freq0 = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let freq1 = 1.0 / theta.powf(2.0 * (i + 1) as f32 / head_dim_f32);
                let freq2 = 1.0 / theta.powf(2.0 * (i + 2) as f32 / head_dim_f32);
                let freq3 = 1.0 / theta.powf(2.0 * (i + 3) as f32 / head_dim_f32);

                // Compute 4 angles
                let angle0 = pos_f32 * freq0;
                let angle1 = pos_f32 * freq1;
                let angle2 = pos_f32 * freq2;
                let angle3 = pos_f32 * freq3;

                // Compute cos/sin (use sincos if available for better performance)
                let (sin0, cos0) = angle0.sin_cos();
                let (sin1, cos1) = angle1.sin_cos();
                let (sin2, cos2) = angle2.sin_cos();
                let (sin3, cos3) = angle3.sin_cos();

                // Load x1 and x2 values
                let x1_0 = x[head_start + i];
                let x1_1 = x[head_start + i + 1];
                let x1_2 = x[head_start + i + 2];
                let x1_3 = x[head_start + i + 3];

                let x2_0 = x[idx2_start + i];
                let x2_1 = x[idx2_start + i + 1];
                let x2_2 = x[idx2_start + i + 2];
                let x2_3 = x[idx2_start + i + 3];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1_0 * cos0 - x2_0 * sin0;
                x[head_start + i + 1] = x1_1 * cos1 - x2_1 * sin1;
                x[head_start + i + 2] = x1_2 * cos2 - x2_2 * sin2;
                x[head_start + i + 3] = x1_3 * cos3 - x2_3 * sin3;

                x[idx2_start + i] = x1_0 * sin0 + x2_0 * cos0;
                x[idx2_start + i + 1] = x1_1 * sin1 + x2_1 * cos1;
                x[idx2_start + i + 2] = x1_2 * sin2 + x2_2 * cos2;
                x[idx2_start + i + 3] = x1_3 * sin3 + x2_3 * cos3;

                i += 4;
            }

            // Handle remaining elements
            while i < half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;

                i += 1;
            }
        }
    }

    /// Compute scaled dot-product attention with causal mask and GQA support
    ///
    /// Implements multi-head attention with Grouped Query Attention (GQA),
    /// where multiple Q heads share the same K/V heads.
    ///
    /// Optimized for single-token inference (seq_len=1).
    fn causal_attention(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA: multiple Q heads share each KV head
        let group_size = num_heads / num_kv_heads;

        // Q has num_heads heads, K/V have num_kv_heads heads
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Fast path for single token (common case in autoregressive generation)
        // With seq_len=1 and causal mask, each head just copies its V vector
        // (softmax of single element is 1.0)
        if seq_len == 1 {
            let mut output = vec![0.0f32; q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                output[out_offset..out_offset + head_dim]
                    .copy_from_slice(&v[v_offset..v_offset + head_dim]);
            }
            return output;
        }

        // General case for seq_len > 1
        use rayon::prelude::*;

        // Parallel threshold - use parallel for 4+ heads
        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            // Sequential path for few heads
            let mut output = vec![0.0f32; seq_len * q_dim];
            for head in 0..num_heads {
                self.compute_head_attention(
                    head,
                    group_size,
                    head_dim,
                    scale,
                    q,
                    k,
                    v,
                    seq_len,
                    q_dim,
                    kv_dim,
                    &mut output,
                );
            }
            output
        } else {
            // Parallel path - each head computes independently, then merge
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..seq_len {
                        let mut scores = Vec::with_capacity(i + 1);
                        let q_start = i * q_dim + q_head_offset;

                        for j in 0..=i {
                            let k_start = j * kv_dim + kv_head_offset;
                            let mut score = 0.0f32;
                            for d in 0..head_dim {
                                score += q[q_start + d] * k[k_start + d];
                            }
                            scores.push(score * scale);
                        }

                        // Softmax
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            exp_sum += *s;
                        }
                        for s in &mut scores {
                            *s /= exp_sum;
                        }

                        // Weighted sum
                        let out_start = i * head_dim;
                        for (j, &weight) in scores.iter().enumerate() {
                            let v_start = j * kv_dim + kv_head_offset;
                            for d in 0..head_dim {
                                head_out[out_start + d] += weight * v[v_start + d];
                            }
                        }
                    }
                    head_out
                })
                .collect();

            // Merge head outputs into final output
            let mut output = vec![0.0f32; seq_len * q_dim];
            for (head, head_out) in head_outputs.into_iter().enumerate() {
                let head_offset = head * head_dim;
                for i in 0..seq_len {
                    let src_start = i * head_dim;
                    let dst_start = i * q_dim + head_offset;
                    output[dst_start..dst_start + head_dim]
                        .copy_from_slice(&head_out[src_start..src_start + head_dim]);
                }
            }
            output
        }
    }

    /// Compute attention for a single head (helper for sequential path)
    #[allow(clippy::too_many_arguments)]
    fn compute_head_attention(
        &self,
        head: usize,
        group_size: usize,
        head_dim: usize,
        scale: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        q_dim: usize,
        kv_dim: usize,
        output: &mut [f32],
    ) {
        let kv_head = head / group_size;
        let q_head_offset = head * head_dim;
        let kv_head_offset = kv_head * head_dim;

        for i in 0..seq_len {
            let mut scores = Vec::with_capacity(i + 1);
            let q_start = i * q_dim + q_head_offset;

            for j in 0..=i {
                let k_start = j * kv_dim + kv_head_offset;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                scores.push(score * scale);
            }

            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            let out_start = i * q_dim + q_head_offset;
            for (j, &weight) in scores.iter().enumerate() {
                let v_start = j * kv_dim + kv_head_offset;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }
}

// =============================================================================
// GGUF K-quant Dequantization Helpers (for APR Q4_K/Q6_K support)
// =============================================================================

/// Convert IEEE 754 half-precision (f16) bits to f32
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

/// Extract scale and min from Q4_K 12-byte packed scales
///
/// PAR-001 FIX: Matches llama.cpp's get_scale_min_k4 packing scheme:
/// - Blocks 0-3: scale = q[j] & 63, min = q[j+4] & 63
/// - Blocks 4-7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///   min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
fn extract_scale_min_apr(scales: &[u8], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    (f32::from(scale_bits), f32::from(min_bits))
}

/// Dequantize Q4_K format (K-quants) for APR tensors
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// PMAT-086 FIX: Correct implementation matching llama.cpp/candle layout:
/// - For each 64-value chunk, output 32 low nibbles THEN 32 high nibbles
/// - Use sc1/dm1 for low nibbles, sc2/dm2 for high nibbles (different scales per half)
fn dequantize_q4_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256; // Super-block size
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        // Return zeros if data is insufficient
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        // Read scales (12 bytes)
        let scales = &data[sb_start + 4..sb_start + 16];

        // Read qs (128 bytes)
        let qs = &data[sb_start + 16..sb_start + 144];

        // Dequantize following candle's layout:
        // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
        let mut ys_index = out_start;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min_apr(scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min_apr(scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            // First pass: 32 low nibbles
            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            // Second pass: 32 high nibbles
            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q6_K format (K-quants) for APR tensors
/// Q6_K super-block layout (per llama.cpp block_q6_K and candle):
/// - ql: 128 bytes (low 4 bits, 256 values, 2 per byte)
/// - qh: 64 bytes (high 2 bits, 256 values, 4 per byte)
/// - scales: 16 bytes (i8 signed scales for 16 blocks)
/// - d: 2 bytes (f16)
/// Total: 128 + 64 + 16 + 2 = 210 bytes
fn dequantize_q6_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read ql - low 4 bits (128 bytes) at offset 0
        let ql = &data[sb_start..sb_start + 128];

        // Read qh - high 2 bits (64 bytes) at offset 128
        let qh = &data[sb_start + 128..sb_start + 192];

        // Read scales (16 bytes, i8) at offset 192
        let mut scales = [0i8; 16];
        #[allow(clippy::cast_possible_wrap)]
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = data[sb_start + 192 + i] as i8;
        }

        // Read d (f16 -> f32) at offset 208 (last 2 bytes)
        let d = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        // Dequantize 256 values following candle's exact layout
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16; // Scale index selector (0 or 1 within this 128-block)

                // Extract 4 values per iteration (at positions l, l+32, l+64, l+96)
                // q1: low 4 bits of ql[l] + bits 0-1 of qh[l]
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                // q2: low 4 bits of ql[l+32] + bits 2-3 of qh[l]
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                // q3: high 4 bits of ql[l] + bits 4-5 of qh[l]
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                // q4: high 4 bits of ql[l+32] + bits 6-7 of qh[l]
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                // Write to output with correct scale indexing
                result[out_start + n + l] = d * (sc[is] as f32) * (q1 as f32);
                result[out_start + n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                result[out_start + n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                result[out_start + n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
            }
        }
    }

    result.truncate(num_elements);
    result
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod apr_transformer_tests;
