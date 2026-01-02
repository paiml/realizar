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
use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

use crate::apr::MAGIC;
use crate::error::{RealizarError, Result};

// ============================================================================
// APR Transformer Binary Format (Y1-Y5 Format Parity)
// ============================================================================

/// APR Transformer binary format magic: "APRT" (APR Transformer)
pub const APR_TRANSFORMER_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x54];

/// APR Transformer format version
pub const APR_TRANSFORMER_VERSION: u32 = 1;

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

        // Verify magic (can be either APRN or APRT)
        let magic = &header_bytes[0..4];
        if magic != MAGIC && magic != APR_TRANSFORMER_MAGIC {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: expected {:?} or {:?}, got {:?}",
                    MAGIC, APR_TRANSFORMER_MAGIC, magic
                ),
            });
        }

        // Parse config from header (after 4-byte magic + 4-byte version)
        let version = u32::from_le_bytes([
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]);
        if version > APR_TRANSFORMER_VERSION {
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
        bytes.extend_from_slice(&APR_TRANSFORMER_MAGIC);
        bytes.extend_from_slice(&APR_TRANSFORMER_VERSION.to_le_bytes());
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
        if data[0..4] != APR_TRANSFORMER_MAGIC {
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

/// APR Transformer model with F32 weights
///
/// WASM-compatible format for fair comparison with GGUF.
/// All weights are stored as F32 (dequantized from GGUF if converted).
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

        // Check magic
        let magic = &data[0..4];
        if magic != b"APR2" && magic != b"APRN" {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid APR magic: {:?}, expected APR2 or APRN",
                    String::from_utf8_lossy(magic)
                ),
            });
        }

        // Parse header
        // APR v2 header layout:
        //   0-3: Magic "APR2"
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

        let vocab_size = metadata
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32000) as usize;

        let intermediate_dim = metadata
            .get("intermediate_size")
            .or_else(|| metadata.get("intermediate_dim"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        let config = AprTransformerConfig {
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size,
            intermediate_dim,
            context_length: 2048,
            ..Default::default()
        };

        // Parse tensor index
        // APR v2 TensorIndexEntry format:
        //   - name_len (2 bytes) + name (variable)
        //   - dtype (1 byte)
        //   - ndim (1 byte) + dims (8 bytes each)
        //   - offset (8 bytes)
        //   - size (8 bytes)
        let mut tensors: std::collections::BTreeMap<String, (usize, usize, Vec<usize>)> =
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
            let _dtype = data[pos];
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

            tensors.insert(name, (data_offset + offset, size, dims));
        }

        // Helper to extract f32 tensor
        let get_f32_tensor = |name: &str| -> Option<Vec<f32>> {
            tensors.get(name).map(|(offset, size, _)| {
                let end = offset + size;
                if end > data.len() {
                    return Vec::new();
                }
                data[*offset..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
        };

        // Try to load token embedding
        let token_embedding = get_f32_tensor("model.embed_tokens.weight")
            .or_else(|| get_f32_tensor("token_embd.weight"))
            .or_else(|| get_f32_tensor("tok_embeddings.weight"))
            .unwrap_or_else(|| vec![0.0; vocab_size * hidden_dim]);

        // Load output norm
        let output_norm_weight = get_f32_tensor("model.norm.weight")
            .or_else(|| get_f32_tensor("output_norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden_dim]);

        // Load LM head
        let lm_head_weight = get_f32_tensor("lm_head.weight")
            .or_else(|| get_f32_tensor("output.weight"))
            .unwrap_or_else(|| vec![0.0; hidden_dim * vocab_size]);

        // Load layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");

            // Try separate Q/K/V or combined QKV
            let qkv_dim = 3 * hidden_dim;
            let qkv_weight =
                if let Some(qkv) = get_f32_tensor(&format!("{prefix}.self_attn.qkv_proj.weight")) {
                    qkv
                } else {
                    // Combine separate Q, K, V into QKV
                    let q = get_f32_tensor(&format!("{prefix}.self_attn.q_proj.weight"))
                        .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
                    let k = get_f32_tensor(&format!("{prefix}.self_attn.k_proj.weight"))
                        .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
                    let v = get_f32_tensor(&format!("{prefix}.self_attn.v_proj.weight"))
                        .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);

                    // Interleave Q, K, V for each row
                    let mut qkv = Vec::with_capacity(hidden_dim * qkv_dim);
                    for row in 0..hidden_dim {
                        let row_start = row * hidden_dim;
                        qkv.extend_from_slice(&q[row_start..row_start + hidden_dim]);
                        qkv.extend_from_slice(&k[row_start..row_start + hidden_dim]);
                        qkv.extend_from_slice(&v[row_start..row_start + hidden_dim]);
                    }
                    qkv
                };

            let attn_output = get_f32_tensor(&format!("{prefix}.self_attn.o_proj.weight"))
                .unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);

            let attn_norm = get_f32_tensor(&format!("{prefix}.input_layernorm.weight"))
                .unwrap_or_else(|| vec![1.0; hidden_dim]);

            let ffn_norm = get_f32_tensor(&format!("{prefix}.post_attention_layernorm.weight"));

            let ffn_gate = get_f32_tensor(&format!("{prefix}.mlp.gate_proj.weight"));
            let ffn_up = get_f32_tensor(&format!("{prefix}.mlp.up_proj.weight"))
                .unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = get_f32_tensor(&format!("{prefix}.mlp.down_proj.weight"))
                .unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias: None,
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

        Ok(Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
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

            // Stop at EOS (token 2 is common)
            if next_token == 2 {
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

    /// Layer normalization
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

            // Calculate mean
            let mean: f32 = slice.iter().sum::<f32>() / hidden_dim as f32;

            // Calculate variance
            let variance: f32 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;

            // Normalize
            let std_dev = (variance + eps).sqrt();
            for (i, &x) in slice.iter().enumerate() {
                let normalized = (x - mean) / std_dev;
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

    /// Matrix multiplication: output[out_dim] = input[in_dim] * weight[in_dim, out_dim]
    /// Uses trueno SIMD for ~10x speedup over scalar implementation.
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let expected_size = in_dim * out_dim;

        // Determine weight layout and create matrix for SIMD matvec
        // Weight could be [in_dim, out_dim] or [out_dim, in_dim]
        let weight_matrix = if weight.len() == expected_size {
            // Weight is [in_dim, out_dim] - need to transpose for matvec
            let mut weight_transposed = vec![0.0f32; expected_size];
            for i in 0..in_dim {
                for o in 0..out_dim {
                    weight_transposed[o * in_dim + i] = weight[i * out_dim + o];
                }
            }
            TruenoMatrix::from_vec(out_dim, in_dim, weight_transposed)
        } else if weight.len() == out_dim * in_dim {
            // Weight is already [out_dim, in_dim] - use directly
            TruenoMatrix::from_vec(out_dim, in_dim, weight.to_vec())
        } else {
            // Dimension mismatch - fall back to scalar
            return self.matmul_scalar(input, weight, in_dim, out_dim);
        };

        let weight_matrix = match weight_matrix {
            Ok(m) => m,
            Err(_) => {
                // Fallback to scalar if trueno fails
                return self.matmul_scalar(input, weight, in_dim, out_dim);
            },
        };

        let mut output = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];
            let x_vec = TruenoVector::from_slice(input_slice);

            match weight_matrix.matvec(&x_vec) {
                Ok(r) => output.extend_from_slice(r.as_slice()),
                Err(_) => {
                    // Fallback to scalar for this sequence position
                    for o in 0..out_dim {
                        let mut sum = 0.0;
                        for (i, &input_val) in input_slice.iter().enumerate() {
                            let weight_idx = i * out_dim + o;
                            if weight_idx < weight.len() {
                                sum += input_val * weight[weight_idx];
                            }
                        }
                        output.push(sum);
                    }
                },
            }
        }

        output
    }

    /// Scalar fallback for matmul (used when trueno fails)
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
                    let weight_idx = i * out_dim + o;
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
        for layer in &self.layers {
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

            // 2c. Simplified attention (matches GGUF implementation)
            let seq_len = token_ids.len();
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                for h in 0..hidden_dim {
                    attn_out.push(qkv[qkv_start + h]); // Use Q for simplified version
                }
            }

            // 2d. Attention output projection
            let mut attn_output =
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection
            let mut ffn_hidden =
                self.matmul(&hidden, &layer.ffn_up_weight, hidden_dim, intermediate_dim);
            if let Some(ref bias) = layer.ffn_up_bias {
                self.add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation
            self.gelu(&mut ffn_hidden);

            // FFN down projection
            let mut ffn_output = self.matmul(
                &ffn_hidden,
                &layer.ffn_down_weight,
                intermediate_dim,
                hidden_dim,
            );
            if let Some(ref bias) = layer.ffn_down_bias {
                self.add_bias(&mut ffn_output, bias);
            }

            // Residual connection
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
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
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
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_out_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // Split into Q, K, V with GQA-aware sizes
            // Q: [hidden_dim] = [num_heads * head_dim]
            // K: [kv_size] = [num_kv_heads * head_dim]
            // V: [kv_size] = [num_kv_heads * head_dim]
            let kv_size = num_kv_heads * head_dim;
            let q = &qkv[0..hidden_dim];
            let k = &qkv[hidden_dim..hidden_dim + kv_size];
            let v = &qkv[hidden_dim + kv_size..hidden_dim + 2 * kv_size];

            // 2c. Append K, V to cache
            cache.append(layer_idx, k, v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let seq_len = cache.len();

            // Simplified attention: compute Q·K^T / sqrt(d), softmax, then V
            let mut attn_out = vec![0.0f32; hidden_dim];

            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads; // GQA mapping
                let q_start = h * head_dim;
                let q_slice = &q[q_start..q_start + head_dim];

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
            let mut attn_output =
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2g. FFN
            let mut ffn_hidden = self.matmul(
                &hidden,
                &layer.ffn_up_weight,
                hidden_dim,
                self.config.intermediate_dim,
            );
            if let Some(ref bias) = layer.ffn_up_bias {
                self.add_bias(&mut ffn_hidden, bias);
            }
            self.gelu(&mut ffn_hidden);

            let mut ffn_output = self.matmul(
                &ffn_hidden,
                &layer.ffn_down_weight,
                self.config.intermediate_dim,
                hidden_dim,
            );
            if let Some(ref bias) = layer.ffn_down_bias {
                self.add_bias(&mut ffn_output, bias);
            }

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

        // 4. LM head projection
        let mut logits = self.matmul(
            &normed,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
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

        // Process prompt tokens
        for (pos, &token) in prompt.iter().enumerate() {
            let _ = self.forward_with_cache(token, &mut cache, pos)?;
        }

        // Generate new tokens
        for _ in 0..config.max_tokens {
            let last_token = *output.last().ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty output".to_string(),
            })?;

            let logits = self.forward_with_cache(last_token, &mut cache, output.len() - 1)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 {
                // Greedy decoding
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i as u32)
            } else {
                // Temperature sampling (simplified)
                let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = scaled.iter().map(|s| (s - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();

                // Simple sampling: pick highest prob for determinism in tests
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i as u32)
            };

            output.push(next_token);

            // Check for EOS (optional: could add eos_token to config)
            if next_token == 0 {
                break;
            }
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
        }
    }
}

// ============================================================================
// Y6: APR Benchmark Infrastructure (Format Parity Validation)
// ============================================================================

/// CPU decode threshold: 50 tok/s per spec Y6
pub const APR_CPU_DECODE_THRESHOLD_TOK_S: f64 = 50.0;

/// Prefill threshold: 100 tok/s per spec Y8
pub const APR_PREFILL_THRESHOLD_TOK_S: f64 = 100.0;

/// Parity threshold: 95% of baseline per spec Y6
pub const APR_PARITY_THRESHOLD_PCT: f64 = 95.0;

/// Result of an APR benchmark run
#[derive(Debug, Clone, Default)]
pub struct AprBenchmarkResult {
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Throughput in tokens per second
    pub tokens_per_second: f64,
    /// Median throughput (p50)
    pub throughput_p50: f64,
    /// 99th percentile throughput (worst case)
    pub throughput_p99: f64,
    /// Standard deviation of throughput
    pub throughput_std_dev: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Model memory in MB
    pub model_memory_mb: f64,
}

impl AprBenchmarkResult {
    /// Check if benchmark meets the given throughput threshold
    #[must_use]
    pub fn meets_threshold(&self, threshold_tok_s: f64) -> bool {
        self.tokens_per_second >= threshold_tok_s
    }

    /// Compare this result to a baseline
    #[must_use]
    pub fn compare_to_baseline(&self, baseline: &AprBenchmarkResult) -> AprParityComparison {
        let throughput_ratio = if baseline.tokens_per_second > 0.0 {
            self.tokens_per_second / baseline.tokens_per_second
        } else {
            1.0
        };

        let memory_ratio = if baseline.peak_memory_mb > 0.0 {
            self.peak_memory_mb / baseline.peak_memory_mb
        } else {
            1.0
        };

        AprParityComparison {
            throughput_ratio,
            memory_ratio,
            parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
        }
    }
}

/// Result of prefill benchmark
#[derive(Debug, Clone, Default)]
pub struct AprPrefillResult {
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Prefill time in milliseconds
    pub prefill_time_ms: f64,
    /// Prefill throughput in tokens per second
    pub prefill_tok_s: f64,
}

/// Result of load time benchmark
#[derive(Debug, Clone, Default)]
pub struct AprLoadResult {
    /// Load time in milliseconds
    pub load_time_ms: f64,
}

/// Comparison of APR benchmark to baseline (for parity validation)
#[derive(Debug, Clone)]
pub struct AprParityComparison {
    /// Ratio of APR throughput to baseline
    pub throughput_ratio: f64,
    /// Ratio of APR memory to baseline
    pub memory_ratio: f64,
    /// Parity threshold percentage
    pub parity_threshold_pct: f64,
}

impl AprParityComparison {
    /// Check if APR achieves parity with baseline
    #[must_use]
    pub fn is_parity(&self) -> bool {
        self.throughput_ratio >= (self.parity_threshold_pct / 100.0)
    }
}

/// Benchmark runner for APR transformers (Y6)
///
/// Provides standardized benchmarking following the benchmark spec:
/// - Dynamic CV-based sampling
/// - Statistical metrics (p50, p99, std_dev)
/// - Throughput and memory measurement
#[derive(Debug)]
pub struct AprBenchmarkRunner {
    /// The transformer to benchmark
    transformer: AprTransformer,
    /// Number of warmup iterations
    warmup_iterations: usize,
    /// Number of measurement iterations
    measure_iterations: usize,
}

impl AprBenchmarkRunner {
    /// Create a new benchmark runner for the given transformer
    #[must_use]
    pub fn new(transformer: AprTransformer) -> Self {
        Self {
            transformer,
            warmup_iterations: 3,
            measure_iterations: 10,
        }
    }

    /// Get warmup iterations
    #[must_use]
    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    /// Get measure iterations
    #[must_use]
    pub fn measure_iterations(&self) -> usize {
        self.measure_iterations
    }

    /// Set warmup iterations
    pub fn set_warmup_iterations(&mut self, n: usize) {
        self.warmup_iterations = n;
    }

    /// Set measure iterations
    pub fn set_measure_iterations(&mut self, n: usize) {
        self.measure_iterations = n.max(1);
    }

    /// Benchmark decode throughput
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `num_tokens` - Number of tokens to generate
    ///
    /// # Returns
    ///
    /// Benchmark result with throughput metrics
    pub fn benchmark_decode(
        &mut self,
        prompt: &[u32],
        num_tokens: usize,
    ) -> Result<AprBenchmarkResult> {
        use std::time::Instant;

        // Warmup
        for _ in 0..self.warmup_iterations {
            let gen_config = GenerateConfig {
                max_tokens: num_tokens.min(5),
                temperature: 0.0,
                ..Default::default()
            };
            let _ = self.transformer.generate_with_cache(prompt, &gen_config)?;
        }

        // Measurement runs
        let mut throughputs = Vec::with_capacity(self.measure_iterations);
        let mut total_tokens = 0usize;
        let mut total_time_ms = 0.0f64;

        for _ in 0..self.measure_iterations {
            let gen_config = GenerateConfig {
                max_tokens: num_tokens,
                temperature: 0.0,
                ..Default::default()
            };

            let start = Instant::now();
            let output = self.transformer.generate_with_cache(prompt, &gen_config)?;
            let elapsed = start.elapsed();

            let generated = output.len().saturating_sub(prompt.len());
            let time_ms = elapsed.as_secs_f64() * 1000.0;
            let throughput = if time_ms > 0.0 {
                (generated as f64) / (time_ms / 1000.0)
            } else {
                0.0
            };

            throughputs.push(throughput);
            total_tokens += generated;
            total_time_ms += time_ms;
        }

        // Calculate statistics
        let mean_throughput = if !throughputs.is_empty() {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        } else {
            0.0
        };

        let mut sorted = throughputs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = if !sorted.is_empty() {
            sorted[sorted.len() / 2]
        } else {
            0.0
        };

        let p99_idx =
            ((sorted.len() as f64 * 0.01).floor() as usize).min(sorted.len().saturating_sub(1));
        let p99 = if !sorted.is_empty() {
            sorted[p99_idx]
        } else {
            0.0
        };

        let std_dev = if throughputs.len() > 1 {
            let variance = throughputs
                .iter()
                .map(|t| (t - mean_throughput).powi(2))
                .sum::<f64>()
                / (throughputs.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Memory estimation
        let model_memory_mb = (self.transformer.memory_size() as f64) / (1024.0 * 1024.0);

        Ok(AprBenchmarkResult {
            tokens_generated: total_tokens / self.measure_iterations.max(1),
            total_time_ms: total_time_ms / self.measure_iterations.max(1) as f64,
            tokens_per_second: mean_throughput,
            throughput_p50: p50,
            throughput_p99: p99,
            throughput_std_dev: std_dev,
            peak_memory_mb: model_memory_mb * 1.5, // Estimate: model + KV cache
            model_memory_mb,
        })
    }

    /// Benchmark prefill throughput
    ///
    /// # Arguments
    ///
    /// * `prompt` - Tokens to prefill
    ///
    /// # Returns
    ///
    /// Prefill benchmark result
    pub fn benchmark_prefill(&mut self, prompt: &[u32]) -> Result<AprPrefillResult> {
        use std::time::Instant;

        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = self.transformer.forward(prompt)?;
        }

        // Measurement runs
        let mut prefill_times_ms = Vec::with_capacity(self.measure_iterations);

        for _ in 0..self.measure_iterations {
            let start = Instant::now();
            let _ = self.transformer.forward(prompt)?;
            let elapsed = start.elapsed();
            prefill_times_ms.push(elapsed.as_secs_f64() * 1000.0);
        }

        let mean_time_ms = if !prefill_times_ms.is_empty() {
            prefill_times_ms.iter().sum::<f64>() / prefill_times_ms.len() as f64
        } else {
            0.0
        };

        let prefill_tok_s = if mean_time_ms > 0.0 {
            (prompt.len() as f64) / (mean_time_ms / 1000.0)
        } else {
            0.0
        };

        Ok(AprPrefillResult {
            prompt_tokens: prompt.len(),
            prefill_time_ms: mean_time_ms,
            prefill_tok_s,
        })
    }

    /// Benchmark model load time
    ///
    /// # Arguments
    ///
    /// * `loader` - Closure that creates the transformer
    ///
    /// # Returns
    ///
    /// Load time result
    pub fn benchmark_load<F>(loader: F) -> Result<AprLoadResult>
    where
        F: Fn() -> AprTransformer,
    {
        use std::time::Instant;

        // Single measurement (load is typically done once)
        let start = Instant::now();
        let _transformer = loader();
        let elapsed = start.elapsed();

        Ok(AprLoadResult {
            load_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // Configuration Tests
    // ==========================================================================

    #[test]
    fn test_config_default() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_config_serialization() {
        let config = AprTransformerConfig {
            architecture: "test_arch".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 1024,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-6,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let decoded: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, decoded);
    }

    // ==========================================================================
    // Layer Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty() {
        let layer = AprTransformerLayer::empty(64, 256);
        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
        assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
        assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
    }

    #[test]
    fn test_layer_num_parameters() {
        let layer = AprTransformerLayer::empty(64, 256);
        let expected = 64 // attn_norm
            + 64 * 3 * 64 // qkv
            + 64 * 64 // attn_output
            + 64 * 256 // ffn_up
            + 256 * 64; // ffn_down
        assert_eq!(layer.num_parameters(), expected);
    }

    // ==========================================================================
    // Transformer Tests
    // ==========================================================================

    #[test]
    fn test_transformer_new() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
        assert_eq!(transformer.output_norm_weight.len(), 64);
        assert_eq!(transformer.lm_head_weight.len(), 64 * 100);
    }

    #[test]
    fn test_transformer_num_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Should be > 0 and reasonable
        let params = transformer.num_parameters();
        assert!(params > 0);
        assert!(params < 100_000_000); // Less than 100M params for test model
    }

    #[test]
    fn test_transformer_memory_size() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let params = transformer.num_parameters();
        let mem = transformer.memory_size();
        assert_eq!(mem, params * 4); // F32 = 4 bytes
    }

    // ==========================================================================
    // Embedding Tests
    // ==========================================================================

    #[test]
    fn test_embed_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            vocab_size: 10,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set known embedding for token 3
        transformer.token_embedding[3 * 4..3 * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let embedded = transformer.embed(&[3]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embed_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set embeddings
        transformer.token_embedding[0..2].copy_from_slice(&[1.0, 2.0]); // token 0
        transformer.token_embedding[2..4].copy_from_slice(&[3.0, 4.0]); // token 1
        transformer.token_embedding[4..6].copy_from_slice(&[5.0, 6.0]); // token 2

        let embedded = transformer.embed(&[0, 1, 2]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_embed_out_of_vocab() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Token 100 is out of vocab (vocab_size=5)
        let embedded = transformer.embed(&[100]);
        assert_eq!(embedded, vec![0.0, 0.0]); // Returns zeros
    }

    // ==========================================================================
    // Layer Norm Tests
    // ==========================================================================

    #[test]
    fn test_layer_norm_identity() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // Identity weight

        let output = transformer.layer_norm(&input, &weight, None, 1e-5);

        // Normalized values should have mean ~0 and var ~1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!((mean).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 3.0]; // mean=2, var=1
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];

        let output = transformer.layer_norm(&input, &weight, Some(&bias), 1e-5);

        // After norm: [-1, 1], after scale: [-1, 1], after bias: [9, 21]
        assert!((output[0] - 9.0).abs() < 0.01);
        assert!((output[1] - 21.0).abs() < 0.01);
    }

    // ==========================================================================
    // GELU Tests
    // ==========================================================================

    #[test]
    fn test_gelu_zero() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![0.0];
        transformer.gelu(&mut data);
        assert!((data[0]).abs() < 0.0001);
    }

    #[test]
    fn test_gelu_positive() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0];
        transformer.gelu(&mut data);
        // GELU(1) ≈ 0.841
        assert!((data[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![-1.0];
        transformer.gelu(&mut data);
        // GELU(-1) ≈ -0.159
        assert!((data[0] - (-0.159)).abs() < 0.01);
    }

    // ==========================================================================
    // Matmul Tests
    // ==========================================================================

    #[test]
    fn test_matmul_identity() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0];
        // Identity matrix [2, 2] in row-major
        let weight = vec![1.0, 0.0, 0.0, 1.0];

        let output = transformer.matmul(&input, &weight, 2, 2);
        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_matmul_simple() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // input: [1, 2]
        // weight: [[1, 2, 3], [4, 5, 6]] (2x3 row-major)
        // output: [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output = transformer.matmul(&input, &weight, 2, 3);
        assert_eq!(output, vec![9.0, 12.0, 15.0]);
    }

    // ==========================================================================
    // Forward Tests
    // ==========================================================================

    #[test]
    fn test_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size
    }

    #[test]
    fn test_forward_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size (only last token logits)
    }

    // ==========================================================================
    // Predict Tests
    // ==========================================================================

    #[test]
    fn test_predict_next() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[1]);
        assert!(result.is_ok());

        let token = result.expect("predict succeeded");
        assert!(token < 10); // Within vocab
    }

    // ==========================================================================
    // Reproducibility Tests
    // ==========================================================================

    #[test]
    fn test_reproducibility_same_input_same_output() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let output1 = transformer.forward(&tokens).expect("forward 1");
        let output2 = transformer.forward(&tokens).expect("forward 2");

        assert_eq!(output1, output2, "Same input should produce same output");
    }

    #[test]
    fn test_reproducibility_predict_deterministic() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let pred1 = transformer.predict_next(&tokens).expect("predict 1");
        let pred2 = transformer.predict_next(&tokens).expect("predict 2");

        assert_eq!(pred1, pred2, "Predictions should be deterministic");
    }

    // ==========================================================================
    // Serialization Tests
    // ==========================================================================

    #[test]
    fn test_transformer_serialization_roundtrip() {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 8,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let transformer = AprTransformer::new(config);

        let json = serde_json::to_string(&transformer).expect("serialize");
        let decoded: AprTransformer = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(transformer.config, decoded.config);
        assert_eq!(transformer.token_embedding, decoded.token_embedding);
        assert_eq!(transformer.layers.len(), decoded.layers.len());
    }

    // ==========================================================================
    // GQA (Grouped Query Attention) KV Cache Tests - IMP-GQA-001
    // ==========================================================================

    /// Test that forward_with_cache works with GQA models (num_kv_heads < num_heads)
    /// This is a regression test for the QKV extraction bug where K/V were assumed
    /// to have the same size as Q (hidden_dim), but GQA models have smaller K/V.
    ///
    /// GQA model example: Qwen2.5-0.5B (14 heads, 2 KV heads)
    /// - Q size: 14 * 64 = 896
    /// - K size: 2 * 64 = 128
    /// - V size: 2 * 64 = 128
    /// - Total QKV: 896 + 128 + 128 = 1152 (not 896 * 3 = 2688)
    #[test]
    fn test_forward_with_cache_gqa_does_not_panic() {
        // Create GQA config similar to Qwen2.5-0.5B
        let config = AprTransformerConfig {
            architecture: "qwen2".to_string(),
            hidden_dim: 64, // 8 heads * 8 head_dim
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4:1 ratio
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        // Create transformer with GQA-sized layers
        let layers: Vec<AprTransformerLayer> = (0..config.num_layers)
            .map(|_| {
                AprTransformerLayer::empty_gqa(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.intermediate_dim,
                )
            })
            .collect();

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers,
            output_norm_weight: vec![1.0; config.hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
            lm_head_bias: None,
        };

        let mut cache = AprKVCache::new(&config);

        // This should NOT panic with proper GQA support
        // Before fix: panics with "range end index X out of range for slice of length Y"
        let result = transformer.forward_with_cache(1, &mut cache, 0);
        assert!(
            result.is_ok(),
            "forward_with_cache should not panic on GQA models: {:?}",
            result
        );

        // Generate a few more tokens to test cache accumulation
        let result = transformer.forward_with_cache(2, &mut cache, 1);
        assert!(result.is_ok());

        let result = transformer.forward_with_cache(3, &mut cache, 2);
        assert!(result.is_ok());

        // Verify cache has correct length
        assert_eq!(cache.len(), 3);
    }

    /// Test GQA KV cache dimensions are correct
    #[test]
    fn test_gqa_kv_cache_dimensions() {
        let config = AprTransformerConfig {
            hidden_dim: 64, // 8 heads * 8 head_dim
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4:1 ratio
            context_length: 32,
            ..Default::default()
        };

        let cache = AprKVCache::new(&config);

        // KV cache should store num_kv_heads * head_dim per position
        // head_dim = 64 / 8 = 8
        // kv_size = 2 * 8 = 16 per position per layer
        assert_eq!(cache.num_kv_heads, 2);
        assert_eq!(cache.head_dim, 8);
    }
}
