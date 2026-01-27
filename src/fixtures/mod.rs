//! Model Fixture Pattern for Test Standardization (Phase 54)
//!
//! Provides RAII-based model fixtures that:
//! - Create valid (or invalid) model files in temporary directories
//! - Provide paths and metadata to tests
//! - Automatically clean up on Drop
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::fixtures::{ModelFixture, ModelConfig};
//!
//! #[test]
//! fn test_model_loading() {
//!     // Setup: creates temp/tiny_llama.gguf
//!     let fixture = ModelFixture::gguf("tiny_llama", ModelConfig::tiny());
//!
//!     // Use the fixture
//!     let model = GGUFModel::from_file(fixture.path()).unwrap();
//!     assert_eq!(model.architecture(), Some("llama"));
//!
//!     // Teardown: automatic on Drop
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::error::{RealizarError, Result};

// =============================================================================
// Model Configuration
// =============================================================================

/// Configuration for generating test models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture name
    pub architecture: String,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length
    pub context_length: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

impl ModelConfig {
    /// Tiny model for fast unit tests (64 hidden, 1 layer)
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Small model for integration tests (256 hidden, 2 layers)
    #[must_use]
    pub fn small() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 256,
            intermediate_dim: 512,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 2,
            vocab_size: 1000,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// GQA model (different num_heads vs num_kv_heads)
    #[must_use]
    pub fn gqa() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 128,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4 Q heads per KV head
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Phi-2 style model (LayerNorm + GELU)
    #[must_use]
    pub fn phi() -> Self {
        Self {
            architecture: "phi".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Qwen style model
    #[must_use]
    pub fn qwen() -> Self {
        Self {
            architecture: "qwen2".to_string(),
            hidden_dim: 128,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 1000000.0,
            eps: 1e-6,
        }
    }

    /// Builder: set architecture
    #[must_use]
    pub fn with_architecture(mut self, arch: &str) -> Self {
        self.architecture = arch.to_string();
        self
    }

    /// Builder: set hidden dimension
    #[must_use]
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Builder: set number of layers
    #[must_use]
    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Builder: set vocab size
    #[must_use]
    pub fn with_vocab_size(mut self, n: usize) -> Self {
        self.vocab_size = n;
        self
    }

    /// Builder: set GQA config
    #[must_use]
    pub fn with_gqa(mut self, num_heads: usize, num_kv_heads: usize) -> Self {
        self.num_heads = num_heads;
        self.num_kv_heads = num_kv_heads;
        self
    }
}

// =============================================================================
// Model Format
// =============================================================================

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// APR format (Aprender native)
    Apr,
    /// SafeTensors format (HuggingFace)
    SafeTensors,
}

impl ModelFormat {
    /// Get file extension for format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Apr => "apr",
            Self::SafeTensors => "safetensors",
        }
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::Apr => write!(f, "APR"),
            Self::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

// =============================================================================
// Model Fixture
// =============================================================================

/// RAII-based model fixture for testing
///
/// Creates a model file in a temporary directory and automatically
/// cleans up when dropped.
pub struct ModelFixture {
    /// Path to the generated model file
    path: PathBuf,
    /// Temporary directory (keeps files alive until Drop)
    _temp_dir: TempDir,
    /// Model format
    format: ModelFormat,
    /// Configuration used to generate the model
    config: ModelConfig,
}

impl ModelFixture {
    // =========================================================================
    // GGUF Fixtures
    // =========================================================================

    /// Create a GGUF model fixture
    ///
    /// # Arguments
    /// * `name` - Model name (used for filename)
    /// * `config` - Model configuration
    ///
    /// # Panics
    /// Panics if fixture creation fails (test should fail fast)
    #[must_use]
    pub fn gguf(name: &str, config: ModelConfig) -> Self {
        Self::try_gguf(name, config).expect("Failed to create GGUF fixture")
    }

    /// Try to create a GGUF model fixture
    pub fn try_gguf(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Generate GGUF data
        let data = Self::generate_gguf_data(&config);

        // Write to file
        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config,
        })
    }

    /// Create a GGUF fixture with invalid magic (for error testing)
    #[must_use]
    pub fn gguf_invalid_magic(name: &str) -> Self {
        Self::try_gguf_invalid_magic(name).expect("Failed to create invalid GGUF fixture")
    }

    /// Try to create a GGUF fixture with invalid magic
    pub fn try_gguf_invalid_magic(name: &str) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Invalid GGUF: wrong magic
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // Invalid magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
        data.resize(32, 0); // Align

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config: ModelConfig::tiny(),
        })
    }

    /// Create a GGUF fixture with unsupported version
    #[must_use]
    pub fn gguf_invalid_version(name: &str) -> Self {
        Self::try_gguf_invalid_version(name).expect("Failed to create invalid version fixture")
    }

    /// Try to create GGUF with invalid version
    pub fn try_gguf_invalid_version(name: &str) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Invalid GGUF: unsupported version
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // Valid magic "GGUF"
        data.extend_from_slice(&999u32.to_le_bytes()); // Invalid version
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.resize(32, 0);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config: ModelConfig::tiny(),
        })
    }

    // =========================================================================
    // SafeTensors Fixtures
    // =========================================================================

    /// Create a SafeTensors model fixture
    #[must_use]
    pub fn safetensors(name: &str, config: ModelConfig) -> Self {
        Self::try_safetensors(name, config).expect("Failed to create SafeTensors fixture")
    }

    /// Try to create a SafeTensors fixture
    pub fn try_safetensors(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.safetensors", name);
        let path = temp_dir.path().join(&filename);

        // Generate SafeTensors data
        let data = Self::generate_safetensors_data(&config);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_safetensors to {}: {}", path.display(), e),
        })?;

        // Generate config.json (required for SafeTensors inference)
        let config_json = format!(
            r#"{{"hidden_size":{},"num_hidden_layers":{},"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"max_position_embeddings":{},"rope_theta":{},"rms_norm_eps":{},"model_type":"{}"}}"#,
            config.hidden_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads,
            config.vocab_size,
            config.intermediate_dim,
            config.context_length,
            config.rope_theta,
            config.eps,
            config.architecture
        );
        let config_path = temp_dir.path().join("config.json");
        fs::write(&config_path, config_json).map_err(|e| RealizarError::IoError {
            message: format!("write_config to {}: {}", config_path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::SafeTensors,
            config,
        })
    }

    // =========================================================================
    // APR Fixtures
    // =========================================================================

    /// Create an APR model fixture
    #[must_use]
    pub fn apr(name: &str, config: ModelConfig) -> Self {
        Self::try_apr(name, config).expect("Failed to create APR fixture")
    }

    /// Try to create an APR fixture
    pub fn try_apr(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.apr", name);
        let path = temp_dir.path().join(&filename);

        // Generate APR v2 data
        let data = Self::generate_apr_data(&config);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_apr to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Apr,
            config,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get path to the model file
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get model format
    #[must_use]
    pub fn format(&self) -> ModelFormat {
        self.format
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Read model file bytes
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        fs::read(&self.path).map_err(|e| RealizarError::IoError {
            message: format!("read_fixture {}: {}", self.path.display(), e),
        })
    }

    // =========================================================================
    // Data Generation (Internal)
    // =========================================================================

    /// Generate valid GGUF data for config
    fn generate_gguf_data(config: &ModelConfig) -> Vec<u8> {
        use crate::gguf::test_factory::GGUFBuilder;

        let arch = &config.architecture;

        // Build minimal GGUF with required tensors
        let builder = GGUFBuilder::new()
            .architecture(arch)
            .hidden_dim(arch, config.hidden_dim as u32)
            .num_layers(arch, config.num_layers as u32)
            .num_heads(arch, config.num_heads as u32)
            .num_kv_heads(arch, config.num_kv_heads as u32)
            .context_length(arch, config.context_length as u32)
            .rope_freq_base(arch, config.rope_theta)
            .rms_epsilon(arch, config.eps)
            .ffn_hidden_dim(arch, config.intermediate_dim as u32);

        // Add token embedding (F32)
        let embed_data: Vec<f32> =
            (0..config.vocab_size * config.hidden_dim).map(|i| (i as f32) * 0.001).collect();
        let builder =
            builder.add_f32_tensor("token_embd.weight", &[config.hidden_dim as u64, config.vocab_size as u64], &embed_data);

        // Add output norm
        let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
        let builder = builder.add_f32_tensor("output_norm.weight", &[config.hidden_dim as u64], &norm_data);

        // Add layer weights (Q4_K for compression)
        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_dim = config.hidden_dim + 2 * kv_dim;

        let mut final_builder = builder;
        for layer in 0..config.num_layers {
            // Attention norm
            let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
            final_builder = final_builder.add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer),
                &[config.hidden_dim as u64],
                &norm_data,
            );

            // QKV (Q4_K)
            let qkv_data = Self::create_q4k_data(config.hidden_dim, qkv_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.attn_qkv.weight", layer),
                &[qkv_dim as u64, config.hidden_dim as u64],
                &qkv_data,
            );

            // Attention output
            let attn_out_data = Self::create_q4k_data(config.hidden_dim, config.hidden_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", layer),
                &[config.hidden_dim as u64, config.hidden_dim as u64],
                &attn_out_data,
            );

            // FFN up
            let ffn_up_data = Self::create_q4k_data(config.hidden_dim, config.intermediate_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", layer),
                &[config.intermediate_dim as u64, config.hidden_dim as u64],
                &ffn_up_data,
            );

            // FFN down
            let ffn_down_data = Self::create_q4k_data(config.intermediate_dim, config.hidden_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", layer),
                &[config.hidden_dim as u64, config.intermediate_dim as u64],
                &ffn_down_data,
            );

            // FFN gate (for SwiGLU)
            if config.architecture == "llama" || config.architecture == "qwen2" {
                let ffn_gate_data = Self::create_q4k_data(config.hidden_dim, config.intermediate_dim);
                final_builder = final_builder.add_q4_k_tensor(
                    &format!("blk.{}.ffn_gate.weight", layer),
                    &[config.intermediate_dim as u64, config.hidden_dim as u64],
                    &ffn_gate_data,
                );
            }

            // FFN norm
            let ffn_norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
            final_builder = final_builder.add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer),
                &[config.hidden_dim as u64],
                &ffn_norm_data,
            );
        }

        // Output/LM head (tied with embeddings in many models)
        let lm_head_data = Self::create_q4k_data(config.hidden_dim, config.vocab_size);
        final_builder = final_builder.add_q4_k_tensor(
            "output.weight",
            &[config.vocab_size as u64, config.hidden_dim as u64],
            &lm_head_data,
        );

        final_builder.build()
    }

    /// Generate SafeTensors data for config
    fn generate_safetensors_data(config: &ModelConfig) -> Vec<u8> {
        use crate::gguf::format_factory::SafetensorsBuilder;

        let mut builder = SafetensorsBuilder::new();

        // Token embedding
        let embed_data: Vec<f32> =
            (0..config.vocab_size * config.hidden_dim).map(|i| (i as f32) * 0.001).collect();
        builder = builder.add_f32_tensor(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_dim],
            &embed_data,
        );

        // Output norm
        let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
        builder = builder.add_f32_tensor("model.norm.weight", &[config.hidden_dim], &norm_data);

        // Layer weights
        for layer in 0..config.num_layers {
            // Attention norm
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.input_layernorm.weight", layer),
                &[config.hidden_dim],
                &norm_data,
            );

            // Q projection
            let q_data: Vec<f32> = vec![0.01; config.hidden_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.q_proj.weight", layer),
                &[config.hidden_dim, config.hidden_dim],
                &q_data,
            );

            // K projection
            let kv_dim = config.num_kv_heads * (config.hidden_dim / config.num_heads);
            let k_data: Vec<f32> = vec![0.01; kv_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.k_proj.weight", layer),
                &[kv_dim, config.hidden_dim],
                &k_data,
            );

            // V projection
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.v_proj.weight", layer),
                &[kv_dim, config.hidden_dim],
                &k_data,
            );

            // O projection
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.o_proj.weight", layer),
                &[config.hidden_dim, config.hidden_dim],
                &q_data,
            );

            // FFN
            let ffn_data: Vec<f32> = vec![0.01; config.hidden_dim * config.intermediate_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.up_proj.weight", layer),
                &[config.intermediate_dim, config.hidden_dim],
                &ffn_data,
            );
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.gate_proj.weight", layer),
                &[config.intermediate_dim, config.hidden_dim],
                &ffn_data,
            );

            let down_data: Vec<f32> = vec![0.01; config.intermediate_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.down_proj.weight", layer),
                &[config.hidden_dim, config.intermediate_dim],
                &down_data,
            );

            // Post attention norm
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.post_attention_layernorm.weight", layer),
                &[config.hidden_dim],
                &norm_data,
            );
        }

        // LM head
        let lm_head_data: Vec<f32> = vec![0.01; config.vocab_size * config.hidden_dim];
        builder = builder.add_f32_tensor(
            "lm_head.weight",
            &[config.vocab_size, config.hidden_dim],
            &lm_head_data,
        );

        builder.build()
    }

    /// Generate APR v2 data for config (PMAT-111: Fixed tensor index serialization)
    ///
    /// APR v2 binary format:
    /// - Header (64 bytes)
    /// - Metadata (JSON, padded to 64 bytes)
    /// - Tensor Index (binary entries)
    /// - Tensor Data (F32 arrays, 64-byte aligned)
    fn generate_apr_data(config: &ModelConfig) -> Vec<u8> {
        // Step 1: Build tensor definitions (name, shape, dtype)
        let mut tensor_defs: Vec<(&str, Vec<usize>, u8)> = Vec::new();
        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        // Global tensors
        tensor_defs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_dim],
            0, // F32
        ));
        tensor_defs.push(("model.norm.weight", vec![config.hidden_dim], 0));
        tensor_defs.push((
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_dim],
            0,
        ));

        // Per-layer tensors
        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");

            // Attention norm
            tensor_defs.push((
                Box::leak(format!("{prefix}.input_layernorm.weight").into_boxed_str()),
                vec![config.hidden_dim],
                0,
            ));

            // Q, K, V, O projections
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.q_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.k_proj.weight").into_boxed_str()),
                vec![kv_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.v_proj.weight").into_boxed_str()),
                vec![kv_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.o_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.hidden_dim],
                0,
            ));

            // FFN norm
            tensor_defs.push((
                Box::leak(
                    format!("{prefix}.post_attention_layernorm.weight").into_boxed_str(),
                ),
                vec![config.hidden_dim],
                0,
            ));

            // FFN projections (gate, up, down)
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.gate_proj.weight").into_boxed_str()),
                vec![config.intermediate_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.up_proj.weight").into_boxed_str()),
                vec![config.intermediate_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.down_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.intermediate_dim],
                0,
            ));
        }

        let tensor_count = tensor_defs.len();

        // Step 2: Calculate tensor data sizes and offsets
        let mut tensor_data_offsets: Vec<u64> = Vec::new();
        let mut tensor_sizes: Vec<u64> = Vec::new();
        let mut current_offset: u64 = 0;

        for (_, shape, dtype) in &tensor_defs {
            let element_count: usize = shape.iter().product();
            let bytes_per_element = match dtype {
                0 => 4, // F32
                1 => 2, // F16
                _ => 4, // Default F32
            };
            let size = (element_count * bytes_per_element) as u64;

            tensor_data_offsets.push(current_offset);
            tensor_sizes.push(size);
            current_offset += size;
            // Align to 64 bytes
            current_offset = current_offset.div_ceil(64) * 64;
        }

        let total_tensor_data_size = current_offset;

        // Step 3: Build the APR file
        let mut data = Vec::new();

        // Header (64 bytes)
        // Magic: "APR\0"
        data.extend_from_slice(b"APR\x00");
        // Version: 2.0
        data.push(2); // major
        data.push(0); // minor
        // Flags: 0
        data.extend_from_slice(&0u16.to_le_bytes());
        // Tensor count
        data.extend_from_slice(&(tensor_count as u32).to_le_bytes());
        // Metadata offset (after header = 64)
        data.extend_from_slice(&64u64.to_le_bytes());

        // Placeholder for metadata size (position 20)
        let metadata_size_offset = data.len();
        data.extend_from_slice(&0u32.to_le_bytes());

        // Placeholder for tensor index offset (position 24)
        let tensor_index_offset_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        // Placeholder for data offset (position 32)
        let data_offset_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        // Checksum (placeholder)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Reserved (pad to 64 bytes)
        data.resize(64, 0);

        // Metadata (JSON)
        let metadata = format!(
            r#"{{"architecture":"{}","hidden_size":{},"num_layers":{},"num_heads":{},"num_kv_heads":{},"vocab_size":{},"intermediate_size":{},"rope_theta":{},"rms_norm_eps":{}}}"#,
            config.architecture,
            config.hidden_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads,
            config.vocab_size,
            config.intermediate_dim,
            config.rope_theta,
            config.eps
        );
        let metadata_bytes = metadata.as_bytes();

        // Update metadata size
        let metadata_size = metadata_bytes.len() as u32;
        data[metadata_size_offset..metadata_size_offset + 4]
            .copy_from_slice(&metadata_size.to_le_bytes());

        data.extend_from_slice(metadata_bytes);

        // Pad to 64-byte boundary
        let padded_len = data.len().div_ceil(64) * 64;
        data.resize(padded_len, 0);

        // Update tensor index offset
        let tensor_index_offset = data.len() as u64;
        data[tensor_index_offset_pos..tensor_index_offset_pos + 8]
            .copy_from_slice(&tensor_index_offset.to_le_bytes());

        // Step 4: Write tensor index entries in binary format
        // Format per TensorEntry::from_binary:
        //   name_len (u16 LE) + name bytes
        //   dtype (u8)
        //   ndim (u8) + dims (u64 LE each)
        //   offset (u64 LE)
        //   size (u64 LE)
        for (i, (name, shape, dtype)) in tensor_defs.iter().enumerate() {
            // Name
            let name_bytes = name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            data.extend_from_slice(name_bytes);

            // Dtype
            data.push(*dtype);

            // Shape: ndim + dims
            data.push(shape.len() as u8);
            for &dim in shape {
                data.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // Offset and size
            data.extend_from_slice(&tensor_data_offsets[i].to_le_bytes());
            data.extend_from_slice(&tensor_sizes[i].to_le_bytes());
        }

        // Pad to 64-byte boundary
        let padded_len = data.len().div_ceil(64) * 64;
        data.resize(padded_len, 0);

        // Update data offset
        let data_offset = data.len() as u64;
        data[data_offset_pos..data_offset_pos + 8].copy_from_slice(&data_offset.to_le_bytes());

        // Step 5: Write tensor data (zeros for synthetic fixture)
        // Using zeros will produce garbage output, but the test will RUN (PMAT-111)
        data.resize(data.len() + total_tensor_data_size as usize, 0);

        data
    }

    /// Create Q4_K data for given dimensions
    fn create_q4k_data(in_dim: usize, out_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        let data_size = out_dim * bytes_per_row;
        let mut data = vec![0u8; data_size];

        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let offset = row * bytes_per_row + sb * 144;
                if offset + 4 <= data.len() {
                    // d=1.0 in f16 format
                    data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
                    // dmin=0
                    data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
                }
            }
        }

        data
    }
}

// =============================================================================
// Falsification Tests (Format×Device Matrix)
// =============================================================================

pub mod falsification_tests;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_gguf_tiny() {
        let fixture = ModelFixture::gguf("test_tiny", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::Gguf);
        assert_eq!(fixture.config().hidden_dim, 64);

        let bytes = fixture.read_bytes().expect("read");
        assert!(!bytes.is_empty());
        // Verify GGUF magic
        assert_eq!(&bytes[0..4], &0x46554747u32.to_le_bytes());
    }

    #[test]
    fn test_fixture_gguf_gqa() {
        let fixture = ModelFixture::gguf("test_gqa", ModelConfig::gqa());
        assert!(fixture.path().exists());
        assert_eq!(fixture.config().num_heads, 8);
        assert_eq!(fixture.config().num_kv_heads, 2);
    }

    #[test]
    fn test_fixture_gguf_invalid_magic() {
        let fixture = ModelFixture::gguf_invalid_magic("bad_magic");
        assert!(fixture.path().exists());

        let bytes = fixture.read_bytes().expect("read");
        // Verify invalid magic
        assert_eq!(&bytes[0..4], &0xDEADBEEFu32.to_le_bytes());
    }

    #[test]
    fn test_fixture_gguf_invalid_version() {
        let fixture = ModelFixture::gguf_invalid_version("bad_version");
        let bytes = fixture.read_bytes().expect("read");
        // Verify version 999
        assert_eq!(&bytes[4..8], &999u32.to_le_bytes());
    }

    #[test]
    fn test_fixture_safetensors() {
        let fixture = ModelFixture::safetensors("test_st", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::SafeTensors);

        let bytes = fixture.read_bytes().expect("read");
        assert!(!bytes.is_empty());
        // SafeTensors starts with JSON header length (u64 LE)
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert!(header_len > 0 && header_len < 1_000_000);
    }

    #[test]
    fn test_fixture_apr() {
        let fixture = ModelFixture::apr("test_apr", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::Apr);

        let bytes = fixture.read_bytes().expect("read");
        // Verify APR magic
        assert_eq!(&bytes[0..4], b"APR\x00");
    }

    #[test]
    fn test_fixture_cleanup_on_drop() {
        let path = {
            let fixture = ModelFixture::gguf("test_cleanup", ModelConfig::tiny());
            let p = fixture.path().to_path_buf();
            assert!(p.exists());
            p
        };
        // After fixture is dropped, file should be cleaned up
        assert!(!path.exists());
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::tiny()
            .with_architecture("qwen2")
            .with_hidden_dim(256)
            .with_layers(4)
            .with_vocab_size(50000)
            .with_gqa(16, 4);

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ModelFormat::Gguf.extension(), "gguf");
        assert_eq!(ModelFormat::Apr.extension(), "apr");
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
    }
}
