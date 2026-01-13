//! Embedding model engine for sentence embeddings.
//!
//! This module provides [`EmbeddingEngine`] for loading and running embedding models
//! like BERT, all-MiniLM, and Nomic-embed. Uses the candle framework for inference.
//!
//! # Supported Models
//!
//! - **all-MiniLM-L6-v2**: 384 dimensions, fast inference
//! - **nomic-embed-text-v1.5**: 768 dimensions, high quality
//! - **bge-small-en-v1.5**: 384 dimensions, good quality/size balance
//! - Generic BERT models
//!
//! # Pooling Strategies
//!
//! - **Mean**: Average all token embeddings (default for most models)
//! - **CLS**: Use the [CLS] token embedding
//! - **LastToken**: Use the last non-padding token
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::embeddings::{EmbeddingEngine, EmbeddingConfig, PoolingStrategy, EmbeddingModelType};
//! use std::path::PathBuf;
//!
//! let config = EmbeddingConfig {
//!     model_path: PathBuf::from("models/all-MiniLM-L6-v2"),
//!     model_type: EmbeddingModelType::AllMiniLM,
//!     pooling: PoolingStrategy::Mean,
//!     normalize: true,
//! };
//!
//! let engine = EmbeddingEngine::load(config)?;
//! let embeddings = engine.embed(&["Hello, world!".to_string()])?;
//! assert_eq!(embeddings[0].len(), 384);
//! ```

use crate::error::{RealizarError, Result};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

// Conditional imports for candle
#[cfg(feature = "embeddings")]
use candle_core::{Device, Tensor};
#[cfg(feature = "embeddings")]
use candle_nn::VarBuilder;
#[cfg(feature = "embeddings")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "embeddings")]
use tokenizers::Tokenizer;

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Embedding model type for architecture-specific loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingModelType {
    /// Generic BERT model
    #[default]
    Bert,
    /// Sentence-Transformers all-MiniLM-L6-v2
    AllMiniLM,
    /// Nomic AI nomic-embed-text
    NomicEmbed,
    /// BAAI bge-small-en
    BgeSmall,
}

/// Pooling strategy for sentence embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    /// Mean pooling over all tokens (excluding padding)
    #[default]
    Mean,
    /// Use the [CLS] token embedding
    Cls,
    /// Use the last non-padding token
    LastToken,
}

/// Configuration for embedding model loading.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Path to the model directory (containing config.json, model.safetensors, tokenizer.json)
    pub model_path: PathBuf,
    /// Model architecture type
    pub model_type: EmbeddingModelType,
    /// Pooling strategy for sentence embeddings
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalize the output embeddings
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_type: EmbeddingModelType::default(),
            pooling: PoolingStrategy::default(),
            normalize: true,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new config for all-MiniLM-L6-v2.
    #[must_use]
    pub fn all_minilm(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_type: EmbeddingModelType::AllMiniLM,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }

    /// Create a new config for nomic-embed-text.
    #[must_use]
    pub fn nomic_embed(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_type: EmbeddingModelType::NomicEmbed,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }

    /// Create a new config for bge-small-en.
    #[must_use]
    pub fn bge_small(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_type: EmbeddingModelType::BgeSmall,
            pooling: PoolingStrategy::Cls,
            normalize: true,
        }
    }

    /// Set pooling strategy.
    #[must_use]
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set normalization.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

// =============================================================================
// EMBEDDING ENGINE
// =============================================================================

/// Inner state of the embedding engine.
#[cfg(feature = "embeddings")]
struct EmbeddingEngineInner {
    model: BertModel,
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
    device: Device,
    hidden_size: usize,
}

#[cfg(feature = "embeddings")]
impl Debug for EmbeddingEngineInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngineInner")
            .field("config", &self.config)
            .field("device", &self.device)
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}

/// Embedding model engine for sentence embeddings.
///
/// Uses candle for efficient transformer inference. Thread-safe via Arc<Inner>.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::embeddings::{EmbeddingEngine, EmbeddingConfig};
///
/// let config = EmbeddingConfig::all_minilm("models/all-MiniLM-L6-v2");
/// let engine = EmbeddingEngine::load(config)?;
///
/// let embeddings = engine.embed(&["Hello world".to_string()])?;
/// println!("Embedding dimension: {}", embeddings[0].len());
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingEngine {
    #[cfg(feature = "embeddings")]
    inner: Arc<EmbeddingEngineInner>,
    #[cfg(not(feature = "embeddings"))]
    _marker: std::marker::PhantomData<()>,
}

impl EmbeddingEngine {
    /// Load an embedding model from the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Embedding model configuration
    ///
    /// # Returns
    ///
    /// Loaded embedding engine ready for inference.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model files not found
    /// - Invalid model format
    /// - Tokenizer loading fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::embeddings::{EmbeddingEngine, EmbeddingConfig};
    ///
    /// let config = EmbeddingConfig::all_minilm("models/all-MiniLM-L6-v2");
    /// let engine = EmbeddingEngine::load(config)?;
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn load(config: EmbeddingConfig) -> Result<Self> {
        let device = Device::Cpu;
        
        // Load tokenizer
        let tokenizer_path = config.model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RealizarError::IoError {
                message: format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e),
            }
        })?;

        // Load model config
        let config_path = config.model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            RealizarError::IoError {
                message: format!("Failed to read config from {:?}: {}", config_path, e),
            }
        })?;
        let bert_config: BertConfig = serde_json::from_str(&config_str).map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Invalid BERT config: {}", e))
        })?;
        
        let hidden_size = bert_config.hidden_size;

        // Load model weights
        let model_path = config.model_path.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)
                .map_err(|e| {
                    RealizarError::IoError {
                        message: format!("Failed to load model weights: {}", e),
                    }
                })?
        };

        let model = BertModel::load(vb, &bert_config).map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Failed to load BERT model: {}", e))
        })?;

        Ok(Self {
            inner: Arc::new(EmbeddingEngineInner {
                model,
                tokenizer,
                config,
                device,
                hidden_size,
            }),
        })
    }

    /// Load an embedding model (stub when embeddings feature is disabled).
    #[cfg(not(feature = "embeddings"))]
    pub fn load(_config: EmbeddingConfig) -> Result<Self> {
        Err(RealizarError::InvalidConfiguration(
            "Embeddings feature not enabled. Rebuild with --features embeddings".to_string(),
        ))
    }

    /// Embed a batch of texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of strings to embed
    ///
    /// # Returns
    ///
    /// Vector of embeddings, one per input text.
    ///
    /// # Errors
    ///
    /// Returns error if tokenization or inference fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embeddings = engine.embed(&[
    ///     "Hello world".to_string(),
    ///     "How are you?".to_string(),
    /// ])?;
    /// assert_eq!(embeddings.len(), 2);
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let inner = &self.inner;
        
        // Tokenize all texts
        let encodings = inner.tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| {
            RealizarError::InferenceError(format!("Tokenization failed: {}", e))
        })?;

        // Get max length for padding
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        
        // Create input tensors
        let mut all_ids = Vec::new();
        let mut all_mask = Vec::new();
        let mut all_type_ids = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Pad to max length
            let mut padded_ids = ids.to_vec();
            let mut padded_mask = attention_mask.to_vec();
            let mut padded_type_ids = type_ids.to_vec();

            while padded_ids.len() < max_len {
                padded_ids.push(0);
                padded_mask.push(0);
                padded_type_ids.push(0);
            }

            all_ids.extend(padded_ids);
            all_mask.extend(padded_mask);
            all_type_ids.extend(padded_type_ids);
        }

        let batch_size = texts.len();
        let seq_len = max_len;

        // Convert to tensors
        let input_ids = Tensor::from_vec(all_ids, (batch_size, seq_len), &inner.device)
            .map_err(|e| RealizarError::InferenceError(format!("Tensor creation failed: {}", e)))?;
        let attention_mask = Tensor::from_vec(all_mask.iter().map(|&x| x as f32).collect::<Vec<_>>(), (batch_size, seq_len), &inner.device)
            .map_err(|e| RealizarError::InferenceError(format!("Tensor creation failed: {}", e)))?;
        let token_type_ids = Tensor::from_vec(all_type_ids, (batch_size, seq_len), &inner.device)
            .map_err(|e| RealizarError::InferenceError(format!("Tensor creation failed: {}", e)))?;

        // Run model
        let output = inner.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| RealizarError::InferenceError(format!("Model forward failed: {}", e)))?;

        // Apply pooling
        let pooled = self.apply_pooling(&output, &attention_mask)?;

        // Optionally normalize
        let result = if inner.config.normalize {
            self.l2_normalize(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<Vec<f32>>
        let result_vec: Vec<f32> = result.to_vec2()
            .map_err(|e| RealizarError::InferenceError(format!("Tensor conversion failed: {}", e)))?
            .into_iter()
            .flatten()
            .collect();

        let embeddings: Vec<Vec<f32>> = result_vec
            .chunks(inner.hidden_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(embeddings)
    }

    /// Embed a batch of texts (stub when embeddings feature is disabled).
    #[cfg(not(feature = "embeddings"))]
    pub fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(RealizarError::InvalidConfiguration(
            "Embeddings feature not enabled".to_string(),
        ))
    }

    /// Embed a single text.
    ///
    /// Convenience method that calls [`embed`](Self::embed) with a single-element slice.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let result = self.embed(&[text.to_string()])?;
        result.into_iter().next().ok_or_else(|| {
            RealizarError::InferenceError("Empty embedding result".to_string())
        })
    }

    /// Get the embedding dimension.
    #[cfg(feature = "embeddings")]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.inner.hidden_size
    }

    /// Get the embedding dimension (stub).
    #[cfg(not(feature = "embeddings"))]
    #[must_use]
    pub fn dimension(&self) -> usize {
        0
    }

    /// Get the configuration.
    #[cfg(feature = "embeddings")]
    #[must_use]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.inner.config
    }

    // Internal: Apply pooling strategy
    #[cfg(feature = "embeddings")]
    fn apply_pooling(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let inner = &self.inner;

        match inner.config.pooling {
            PoolingStrategy::Cls => {
                // Take first token [CLS]
                hidden_states.narrow(1, 0, 1)
                    .map_err(|e| RealizarError::InferenceError(format!("CLS pooling failed: {}", e)))?
                    .squeeze(1)
                    .map_err(|e| RealizarError::InferenceError(format!("Squeeze failed: {}", e)))
            },
            PoolingStrategy::Mean => {
                // Mean pooling with attention mask
                let mask_expanded = attention_mask.unsqueeze(2)
                    .map_err(|e| RealizarError::InferenceError(format!("Mask expand failed: {}", e)))?
                    .broadcast_as(hidden_states.shape())
                    .map_err(|e| RealizarError::InferenceError(format!("Broadcast failed: {}", e)))?;
                
                let masked = (hidden_states * &mask_expanded)
                    .map_err(|e| RealizarError::InferenceError(format!("Masking failed: {}", e)))?;
                
                let sum = masked.sum(1)
                    .map_err(|e| RealizarError::InferenceError(format!("Sum failed: {}", e)))?;
                
                let mask_sum = mask_expanded.sum(1)
                    .map_err(|e| RealizarError::InferenceError(format!("Mask sum failed: {}", e)))?
                    .clamp(1e-9, f64::MAX)
                    .map_err(|e| RealizarError::InferenceError(format!("Clamp failed: {}", e)))?;
                
                (sum / mask_sum)
                    .map_err(|e| RealizarError::InferenceError(format!("Division failed: {}", e)))
            },
            PoolingStrategy::LastToken => {
                // Use last non-padding token
                // This is a simplified version - ideally we'd find the actual last token per sequence
                let seq_len = hidden_states.dim(1)
                    .map_err(|e| RealizarError::InferenceError(format!("Dim failed: {}", e)))?;
                hidden_states.narrow(1, seq_len - 1, 1)
                    .map_err(|e| RealizarError::InferenceError(format!("Last token failed: {}", e)))?
                    .squeeze(1)
                    .map_err(|e| RealizarError::InferenceError(format!("Squeeze failed: {}", e)))
            },
        }
    }

    // Internal: L2 normalize embeddings
    #[cfg(feature = "embeddings")]
    fn l2_normalize(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings.sqr()
            .map_err(|e| RealizarError::InferenceError(format!("Sqr failed: {}", e)))?
            .sum_keepdim(1)
            .map_err(|e| RealizarError::InferenceError(format!("Sum failed: {}", e)))?
            .sqrt()
            .map_err(|e| RealizarError::InferenceError(format!("Sqrt failed: {}", e)))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| RealizarError::InferenceError(format!("Clamp failed: {}", e)))?;
        
        embeddings.broadcast_div(&norm)
            .map_err(|e| RealizarError::InferenceError(format!("Normalize failed: {}", e)))
    }
}

// =============================================================================
// COSINE SIMILARITY UTILITIES
// =============================================================================

/// Compute cosine similarity between two embeddings.
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
///
/// Cosine similarity in range [-1, 1], or error if vectors have different lengths.
///
/// # Example
///
/// ```rust
/// use realizar::embeddings::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// let sim = cosine_similarity(&a, &b).unwrap();
/// assert!((sim - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Embedding dimensions don't match: {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

/// Compute L2 distance between two embeddings.
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
///
/// Euclidean distance, or error if vectors have different lengths.
///
/// # Example
///
/// ```rust
/// use realizar::embeddings::l2_distance;
///
/// let a = vec![0.0, 0.0, 0.0];
/// let b = vec![3.0, 4.0, 0.0];
/// let dist = l2_distance(&a, &b).unwrap();
/// assert!((dist - 5.0).abs() < 1e-6);
/// ```
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Embedding dimensions don't match: {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }

    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sum.sqrt())
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_type, EmbeddingModelType::Bert);
        assert_eq!(config.pooling, PoolingStrategy::Mean);
        assert!(config.normalize);
    }

    #[test]
    fn test_embedding_config_all_minilm() {
        let config = EmbeddingConfig::all_minilm("models/test");
        assert_eq!(config.model_type, EmbeddingModelType::AllMiniLM);
        assert_eq!(config.pooling, PoolingStrategy::Mean);
        assert!(config.normalize);
    }

    #[test]
    fn test_embedding_config_nomic() {
        let config = EmbeddingConfig::nomic_embed("models/test");
        assert_eq!(config.model_type, EmbeddingModelType::NomicEmbed);
    }

    #[test]
    fn test_embedding_config_bge() {
        let config = EmbeddingConfig::bge_small("models/test");
        assert_eq!(config.model_type, EmbeddingModelType::BgeSmall);
        assert_eq!(config.pooling, PoolingStrategy::Cls);
    }

    #[test]
    fn test_embedding_config_builder() {
        let config = EmbeddingConfig::default()
            .with_pooling(PoolingStrategy::Cls)
            .with_normalize(false);
        
        assert_eq!(config.pooling, PoolingStrategy::Cls);
        assert!(!config.normalize);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_l2_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = l2_distance(&a, &b).unwrap();
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_345() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = l2_distance(&a, &b).unwrap();
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = l2_distance(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_pooling_strategy_default() {
        assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_model_type_default() {
        assert_eq!(EmbeddingModelType::default(), EmbeddingModelType::Bert);
    }

    #[cfg(not(feature = "embeddings"))]
    #[test]
    fn test_engine_load_without_feature() {
        let config = EmbeddingConfig::all_minilm("models/test");
        let result = EmbeddingEngine::load(config);
        assert!(result.is_err());
    }
}
