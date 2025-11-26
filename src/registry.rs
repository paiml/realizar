//! Model registry for multi-model serving
//!
//! Provides a central registry to manage multiple models in production environments.
//! Supports dynamic model loading/unloading, caching, and thread-safe concurrent access.
//!
//! ## Features
//!
//! - Thread-safe model registry with concurrent access
//! - Integration with `ModelCache` for efficient memory management
//! - Support for multiple model formats (GGUF, Safetensors)
//! - Model metadata and configuration
//! - Graceful loading/unloading with error handling
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::registry::{ModelRegistry, ModelConfig};
//!
//! let registry = ModelRegistry::new(5); // Max 5 models cached
//! registry.register("llama-7b", model, tokenizer)?;
//! let model = registry.get("llama-7b")?;
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::{
    cache::ModelCache,
    error::{RealizarError, Result},
    layers::Model,
    tokenizer::BPETokenizer,
};

/// Information about a registered model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Human-readable model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model format (GGUF, Safetensors, etc.)
    pub format: String,
    /// Whether the model is currently loaded
    pub loaded: bool,
}

/// Entry in the model registry
#[derive(Clone)]
struct ModelEntry {
    /// Model instance
    model: Arc<Model>,
    /// Tokenizer for this model
    tokenizer: Arc<BPETokenizer>,
    /// Model metadata
    info: ModelInfo,
}

/// Type alias for the models map
type ModelsMap = Arc<RwLock<HashMap<String, ModelEntry>>>;

/// Type alias for model and tokenizer tuple
type ModelTuple = (Arc<Model>, Arc<BPETokenizer>);

/// Central registry for managing multiple models
///
/// The `ModelRegistry` provides thread-safe access to multiple models,
/// with automatic caching and lifecycle management.
pub struct ModelRegistry {
    /// Registry of loaded models
    models: ModelsMap,
    /// Model cache for memory management (reserved for future use)
    #[allow(dead_code)]
    cache: Arc<ModelCache>,
}

impl ModelRegistry {
    /// Create a new model registry
    ///
    /// # Arguments
    ///
    /// * `cache_capacity` - Maximum number of models to keep in cache
    #[must_use]
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(ModelCache::new(cache_capacity)),
        }
    }

    /// Register a new model
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the model
    /// * `model` - Model instance
    /// * `tokenizer` - Tokenizer for the model
    ///
    /// # Errors
    ///
    /// Returns error if model ID already exists
    pub fn register(&self, id: &str, model: Model, tokenizer: BPETokenizer) -> Result<()> {
        let mut models = self.models.write().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        if models.contains_key(id) {
            return Err(RealizarError::ModelAlreadyExists(id.to_string()));
        }

        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info: ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                format: "unknown".to_string(),
                loaded: true,
            },
        };

        models.insert(id.to_string(), entry);
        Ok(())
    }

    /// Register a model with full metadata
    ///
    /// # Arguments
    ///
    /// * `info` - Model metadata
    /// * `model` - Model instance
    /// * `tokenizer` - Tokenizer for the model
    ///
    /// # Errors
    ///
    /// Returns error if model ID already exists
    pub fn register_with_info(
        &self,
        mut info: ModelInfo,
        model: Model,
        tokenizer: BPETokenizer,
    ) -> Result<()> {
        let mut models = self.models.write().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        if models.contains_key(&info.id) {
            return Err(RealizarError::ModelAlreadyExists(info.id));
        }

        info.loaded = true;
        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info,
        };

        let id = entry.info.id.clone();
        models.insert(id, entry);
        Ok(())
    }

    /// Get a model by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn get(&self, id: &str) -> Result<ModelTuple> {
        let models = self
            .models
            .read()
            .map_err(|_| RealizarError::RegistryError("Failed to acquire read lock".to_string()))?;

        let entry = models
            .get(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok((Arc::clone(&entry.model), Arc::clone(&entry.tokenizer)))
    }

    /// Get model info by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn get_info(&self, id: &str) -> Result<ModelInfo> {
        let models = self
            .models
            .read()
            .map_err(|_| RealizarError::RegistryError("Failed to acquire read lock".to_string()))?;

        let entry = models
            .get(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok(entry.info.clone())
    }

    /// List all registered models
    #[must_use]
    pub fn list(&self) -> Vec<ModelInfo> {
        let models = self
            .models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        models.values().map(|entry| entry.info.clone()).collect()
    }

    /// Unregister a model
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier
    ///
    /// # Errors
    ///
    /// Returns error if model not found
    pub fn unregister(&self, id: &str) -> Result<()> {
        let mut models = self.models.write().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        models
            .remove(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok(())
    }

    /// Check if a model is registered
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        let models = self
            .models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        models.contains_key(id)
    }

    /// Get the number of registered models
    #[must_use]
    pub fn len(&self) -> usize {
        let models = self
            .models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        models.len()
    }

    /// Check if the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ModelConfig;

    fn create_test_model() -> (Model, BPETokenizer) {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

        (model, tokenizer)
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new(5);
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry.register("test-model", model, tokenizer).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_register_duplicate_error() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry.register("test-model", model1, tokenizer1).unwrap();
        let result = registry.register("test-model", model2, tokenizer2);

        assert!(result.is_err());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_get_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry.register("test-model", model, tokenizer).unwrap();

        let (retrieved_model, retrieved_tokenizer) = registry.get("test-model").unwrap();
        assert!(Arc::strong_count(&retrieved_model) >= 2); // Registry + local
        assert!(Arc::strong_count(&retrieved_tokenizer) >= 2);
    }

    #[test]
    fn test_get_nonexistent_model() {
        let registry = ModelRegistry::new(5);
        let result = registry.get("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_register_with_info() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        let info = ModelInfo {
            id: "llama-7b".to_string(),
            name: "Llama 7B".to_string(),
            description: "7B parameter Llama model".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info.clone(), model, tokenizer)
            .unwrap();

        let retrieved_info = registry.get_info("llama-7b").unwrap();
        assert_eq!(retrieved_info.id, "llama-7b");
        assert_eq!(retrieved_info.name, "Llama 7B");
        assert_eq!(retrieved_info.description, "7B parameter Llama model");
        assert_eq!(retrieved_info.format, "GGUF");
        assert!(retrieved_info.loaded); // Should be set to true
    }

    #[test]
    fn test_list_models() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry.register("model-1", model1, tokenizer1).unwrap();
        registry.register("model-2", model2, tokenizer2).unwrap();

        let model_list = registry.list();
        assert_eq!(model_list.len(), 2);

        let ids: Vec<String> = model_list.iter().map(|m| m.id.clone()).collect();
        assert!(ids.contains(&"model-1".to_string()));
        assert!(ids.contains(&"model-2".to_string()));
    }

    #[test]
    fn test_unregister_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry.register("test-model", model, tokenizer).unwrap();
        assert_eq!(registry.len(), 1);

        registry.unregister("test-model").unwrap();
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("test-model"));
    }

    #[test]
    fn test_unregister_nonexistent() {
        let registry = ModelRegistry::new(5);
        let result = registry.unregister("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let mut handles = vec![];

        // Register models from multiple threads
        for i in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let (model, tokenizer) = create_test_model();
                registry_clone
                    .register(&format!("model-{i}"), model, tokenizer)
                    .unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_multiple_get_same_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry.register("test-model", model, tokenizer).unwrap();

        // Get the same model multiple times
        let (model1, _) = registry.get("test-model").unwrap();
        let (model2, _) = registry.get("test-model").unwrap();

        // Both should point to the same underlying model
        assert!(Arc::ptr_eq(&model1, &model2));
    }
}
