//! Model Fixtures for Testing
//!
//! Concrete fixture implementations for each model format.
//! Inspired by PyTorch's `ModuleInfo` / `ModuleInput` pattern.

use super::{
    generators::{ModelWeights, SyntheticWeightGenerator},
    ConstructorInput, Device, ModelConfig, ModelFormat, QuantType,
};
use crate::error::RealizarError;
use crate::Result;

/// Trait for model fixtures that can be tested
pub trait ModelFixture: Send + Sync {
    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Get model format
    fn format(&self) -> ModelFormat;

    /// Get quantization type
    fn quant_type(&self) -> QuantType;

    /// Run forward pass on specified device
    fn forward(&self, device: Device, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Get embedding for a token
    fn embed(&self, device: Device, token: u32) -> Result<Vec<f32>>;

    /// Serialize to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Convert to another format
    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>>;

    /// Memory footprint in bytes
    fn memory_bytes(&self) -> usize;
}

/// GGUF format fixture
pub struct GgufFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl GgufFixture {
    /// Create from configuration and seed
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, quant);
        Self { config, weights }
    }

    /// Create tiny GQA model for fast tests
    pub fn tiny_gqa() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::Q4_0, 42)
    }

    /// Create tiny MHA model (no GQA)
    pub fn tiny_mha() -> Self {
        let mut config = ModelConfig::tiny();
        config.num_kv_heads = config.num_heads; // MHA
        Self::new(config, QuantType::Q4_0, 42)
    }

    /// Create small model for integration tests
    pub fn small() -> Self {
        Self::new(ModelConfig::small(), QuantType::Q4_K, 42)
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

impl ModelFixture for GgufFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::GGUF
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        // Placeholder - actual implementation would use GGUFTransformer
        // For now, return deterministic output based on config
        let output_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; output_size];

        // Simple deterministic logits based on token values
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) / (output_size as f32) - 0.5) * 2.0;
        }

        Ok(logits)
    }

    fn embed(&self, _device: Device, token: u32) -> Result<Vec<f32>> {
        // Return deterministic embedding
        let mut embedding = vec![0.0f32; self.config.hidden_dim];
        let scale = 1.0 / (self.config.hidden_dim as f32).sqrt();

        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((token as usize * 17 + i * 13) % 1000) as f32 / 1000.0 * scale;
        }

        Ok(embedding)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Build minimal GGUF format
        let mut bytes = Vec::new();

        // GGUF magic: "GGUF"
        bytes.extend_from_slice(b"GGUF");

        // Version (3)
        bytes.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count
        let tensor_count = 1 + self.config.num_layers * 9 + 2; // embed + layers + output
        bytes.extend_from_slice(&(tensor_count as u64).to_le_bytes());

        // Metadata KV count (simplified)
        bytes.extend_from_slice(&10u64.to_le_bytes());

        // Add key metadata
        // num_heads
        write_gguf_kv(
            &mut bytes,
            "llama.attention.head_count",
            self.config.num_heads as u32,
        );
        // num_kv_heads
        write_gguf_kv(
            &mut bytes,
            "llama.attention.head_count_kv",
            self.config.num_kv_heads as u32,
        );
        // hidden_dim
        write_gguf_kv(
            &mut bytes,
            "llama.embedding_length",
            self.config.hidden_dim as u32,
        );
        // num_layers
        write_gguf_kv(
            &mut bytes,
            "llama.block_count",
            self.config.num_layers as u32,
        );
        // vocab_size
        write_gguf_kv(
            &mut bytes,
            "llama.vocab_size",
            self.config.vocab_size as u32,
        );

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::APR => Ok(Box::new(AprFixture::from_gguf(self)?)),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture::from_gguf(self)?)),
            ModelFormat::GGUF => {
                // Clone self
                Ok(Box::new(GgufFixture {
                    config: self.config.clone(),
                    weights: self.weights.clone(),
                }))
            },
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "GGUF to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// APR format fixture
pub struct AprFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl AprFixture {
    /// Create from configuration
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, quant);
        Self { config, weights }
    }

    /// Create tiny GQA model
    pub fn tiny_gqa() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::Q4_0, 42)
    }

    /// Create from GGUF fixture
    pub fn from_gguf(gguf: &GgufFixture) -> Result<Self> {
        Ok(Self {
            config: gguf.config.clone(),
            weights: gguf.weights.clone(),
        })
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

impl ModelFixture for AprFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::APR
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        // Same placeholder as GGUF
        let output_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; output_size];

        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) / (output_size as f32) - 0.5) * 2.0;
        }

        Ok(logits)
    }

    fn embed(&self, _device: Device, token: u32) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0f32; self.config.hidden_dim];
        let scale = 1.0 / (self.config.hidden_dim as f32).sqrt();

        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((token as usize * 17 + i * 13) % 1000) as f32 / 1000.0 * scale;
        }

        Ok(embedding)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // APR magic: "APR\x02" (version 2)
        bytes.extend_from_slice(b"APR\x02");

        // Header size
        bytes.extend_from_slice(&64u32.to_le_bytes());

        // Metadata as JSON
        let metadata = serde_json::json!({
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "hidden_size": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "vocab_size": self.config.vocab_size,
            "intermediate_size": self.config.intermediate_dim,
            "rope_theta": self.config.rope_theta,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("APR metadata serialization failed: {}", e),
            })?;

        // Metadata offset and size
        bytes.extend_from_slice(&64u64.to_le_bytes()); // offset
        bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // size

        // Pad header to 64 bytes
        while bytes.len() < 64 {
            bytes.push(0);
        }

        // Append metadata
        bytes.extend_from_slice(&metadata_bytes);

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::GGUF => Ok(Box::new(GgufFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::APR => Ok(Box::new(AprFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "APR to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// Safetensors format fixture
pub struct SafetensorsFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl SafetensorsFixture {
    /// Create from configuration
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        // Safetensors only supports F32/F16/BF16
        let actual_quant = if quant.supported_by(ModelFormat::Safetensors) {
            quant
        } else {
            QuantType::F32
        };

        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, actual_quant);
        Self { config, weights }
    }

    /// Create tiny model
    pub fn tiny() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::F32, 42)
    }

    /// Create from GGUF fixture
    pub fn from_gguf(gguf: &GgufFixture) -> Result<Self> {
        // Dequantize if needed
        let quant = if gguf.quant_type().supported_by(ModelFormat::Safetensors) {
            gguf.quant_type()
        } else {
            QuantType::F32
        };

        let gen = SyntheticWeightGenerator::new(42);
        let weights = gen.generate_model_weights(&gguf.config, quant);

        Ok(Self {
            config: gguf.config.clone(),
            weights,
        })
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

impl ModelFixture for SafetensorsFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::Safetensors
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        let output_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; output_size];

        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) / (output_size as f32) - 0.5) * 2.0;
        }

        Ok(logits)
    }

    fn embed(&self, _device: Device, token: u32) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0f32; self.config.hidden_dim];
        let scale = 1.0 / (self.config.hidden_dim as f32).sqrt();

        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((token as usize * 17 + i * 13) % 1000) as f32 / 1000.0 * scale;
        }

        Ok(embedding)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Simplified safetensors format
        let header = serde_json::json!({
            "__metadata__": {
                "format": "pt",
                "num_heads": self.config.num_heads,
                "num_kv_heads": self.config.num_kv_heads,
            },
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [self.config.vocab_size, self.config.hidden_dim],
                "data_offsets": [0, self.weights.embed_weights.len()]
            }
        });

        let header_bytes = serde_json::to_vec(&header).map_err(|e| RealizarError::FormatError {
            reason: format!("Safetensors header serialization failed: {}", e),
        })?;
        let header_len = header_bytes.len() as u64;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&self.weights.embed_weights);

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::GGUF => Ok(Box::new(GgufFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::APR => Ok(Box::new(AprFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "Safetensors to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// Create fixture from format and constructor input
pub fn create_fixture(
    format: ModelFormat,
    input: &ConstructorInput,
) -> Result<Box<dyn ModelFixture>> {
    match format {
        ModelFormat::GGUF => Ok(Box::new(GgufFixture::from_constructor(input))),
        ModelFormat::APR => Ok(Box::new(AprFixture::from_constructor(input))),
        ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture::from_constructor(input))),
        ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
            operation: "create_fixture".to_string(),
            reason: "PyTorch fixtures not yet implemented".to_string(),
        }),
    }
}

/// Helper to write GGUF key-value pair
fn write_gguf_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    // Key length
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    // Key bytes
    bytes.extend_from_slice(key.as_bytes());
    // Value type (4 = U32)
    bytes.extend_from_slice(&4u32.to_le_bytes());
    // Value
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_fixture_creation() {
        let fixture = GgufFixture::tiny_gqa();
        assert_eq!(fixture.config().num_heads, 4);
        assert_eq!(fixture.config().num_kv_heads, 2);
        assert!(fixture.config().is_gqa());
    }

    #[test]
    fn test_gguf_to_apr_conversion() {
        let gguf = GgufFixture::tiny_gqa();
        let apr = gguf.convert_to(ModelFormat::APR).unwrap();

        assert_eq!(apr.format(), ModelFormat::APR);
        assert_eq!(apr.config().num_heads, gguf.config().num_heads);
        assert_eq!(apr.config().num_kv_heads, gguf.config().num_kv_heads);
    }

    #[test]
    fn test_gguf_round_trip() {
        let original = GgufFixture::tiny_gqa();
        let apr = original.convert_to(ModelFormat::APR).unwrap();
        let back = apr.convert_to(ModelFormat::GGUF).unwrap();

        assert_eq!(back.config().num_heads, original.config().num_heads);
        assert_eq!(back.config().num_kv_heads, original.config().num_kv_heads);
        assert_eq!(back.config().hidden_dim, original.config().hidden_dim);
    }

    #[test]
    fn test_safetensors_quant_fallback() {
        // Q4_K is not supported by Safetensors, should fallback to F32
        let fixture = SafetensorsFixture::new(ModelConfig::tiny(), QuantType::Q4_K, 42);
        assert_eq!(fixture.quant_type(), QuantType::F32);
    }

    #[test]
    fn test_fixture_serialization() {
        let gguf = GgufFixture::tiny_gqa();
        let bytes = gguf.to_bytes().unwrap();

        // Should start with GGUF magic
        assert_eq!(&bytes[0..4], b"GGUF");
    }

    #[test]
    fn test_apr_serialization() {
        let apr = AprFixture::tiny_gqa();
        let bytes = apr.to_bytes().unwrap();

        // Should start with APR magic
        assert_eq!(&bytes[0..4], b"APR\x02");
    }

    #[test]
    fn test_forward_output_size() {
        let fixture = GgufFixture::tiny_gqa();
        let tokens = vec![1, 2, 3];
        let output = fixture.forward(Device::Cpu, &tokens).unwrap();

        assert_eq!(output.len(), fixture.config().vocab_size);
    }

    #[test]
    fn test_embed_output_size() {
        let fixture = GgufFixture::tiny_gqa();
        let embedding = fixture.embed(Device::Cpu, 42).unwrap();

        assert_eq!(embedding.len(), fixture.config().hidden_dim);
    }

    #[test]
    fn test_create_fixture_dispatch() {
        let input = ConstructorInput::new(ModelConfig::tiny());

        let gguf = create_fixture(ModelFormat::GGUF, &input).unwrap();
        assert_eq!(gguf.format(), ModelFormat::GGUF);

        let apr = create_fixture(ModelFormat::APR, &input).unwrap();
        assert_eq!(apr.format(), ModelFormat::APR);
    }

    #[test]
    fn test_memory_bytes() {
        let fixture = GgufFixture::tiny_gqa();
        let bytes = fixture.memory_bytes();

        // Should be non-zero
        assert!(bytes > 0);

        // Larger config should use more memory
        let large = GgufFixture::new(ModelConfig::small(), QuantType::Q4_0, 42);
        assert!(large.memory_bytes() > bytes);
    }

    #[test]
    fn test_gguf_tiny_mha() {
        let fixture = GgufFixture::tiny_mha();
        assert!(!fixture.config().is_gqa());
        assert_eq!(fixture.config().num_kv_heads, fixture.config().num_heads);
    }

    #[test]
    fn test_gguf_small() {
        let fixture = GgufFixture::small();
        assert_eq!(fixture.config().hidden_dim, 256);
    }

    #[test]
    fn test_gguf_to_safetensors_conversion() {
        let gguf = GgufFixture::tiny_gqa();
        let st = gguf.convert_to(ModelFormat::Safetensors).unwrap();
        assert_eq!(st.format(), ModelFormat::Safetensors);
    }

    #[test]
    fn test_unsupported_conversions() {
        let gguf = GgufFixture::tiny_gqa();
        assert!(gguf.convert_to(ModelFormat::PyTorch).is_err());

        let apr = AprFixture::tiny_gqa();
        assert!(apr.convert_to(ModelFormat::PyTorch).is_err());

        let st = SafetensorsFixture::tiny();
        assert!(st.convert_to(ModelFormat::PyTorch).is_err());
    }

    #[test]
    fn test_clone_conversions() {
        let gguf = GgufFixture::tiny_gqa();
        let gguf2 = gguf.convert_to(ModelFormat::GGUF).unwrap();
        assert_eq!(gguf2.format(), ModelFormat::GGUF);

        let apr = AprFixture::tiny_gqa();
        let apr2 = apr.convert_to(ModelFormat::APR).unwrap();
        assert_eq!(apr2.format(), ModelFormat::APR);

        let st = SafetensorsFixture::tiny();
        let st2 = st.convert_to(ModelFormat::Safetensors).unwrap();
        assert_eq!(st2.format(), ModelFormat::Safetensors);
    }

    #[test]
    fn test_safetensors_serialization() {
        let st = SafetensorsFixture::tiny();
        let bytes = st.to_bytes().unwrap();
        assert!(bytes.len() > 8);
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert!(header_len > 0);
    }

    #[test]
    fn test_create_fixture_unsupported() {
        let input = ConstructorInput::new(ModelConfig::tiny());
        assert!(create_fixture(ModelFormat::PyTorch, &input).is_err());
    }

    #[test]
    fn test_apr_to_safetensors_conversion() {
        let apr = AprFixture::tiny_gqa();
        let st = apr.convert_to(ModelFormat::Safetensors).unwrap();
        assert_eq!(st.format(), ModelFormat::Safetensors);
    }
}
