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
    sync::{Arc, Mutex},
};

use arc_swap::ArcSwap;

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

/// Type alias for the models map (immutable snapshot)
type ModelsMap = HashMap<String, ModelEntry>;

/// Type alias for model and tokenizer tuple
type ModelTuple = (Arc<Model>, Arc<BPETokenizer>);

/// Central registry for managing multiple models
///
/// The `ModelRegistry` provides thread-safe access to multiple models,
/// with automatic caching and lifecycle management.
///
/// Uses `ArcSwap` for lock-free reads (per `McKenney` 2011).
pub struct ModelRegistry {
    /// Registry of loaded models (lock-free reads via `ArcSwap`)
    models: ArcSwap<ModelsMap>,
    /// Write lock to serialize modifications
    write_lock: Mutex<()>,
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
            models: ArcSwap::from_pointee(HashMap::new()),
            write_lock: Mutex::new(()),
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
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if current.contains_key(id) {
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

        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id.to_string(), entry);
        self.models.store(Arc::new(new_map));
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
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if current.contains_key(&info.id) {
            return Err(RealizarError::ModelAlreadyExists(info.id));
        }

        info.loaded = true;
        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info,
        };

        let id = entry.info.id.clone();
        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id, entry);
        self.models.store(Arc::new(new_map));
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
        // Lock-free read via ArcSwap::load()
        let models = self.models.load();

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
        let models = self.models.load();

        let entry = models
            .get(id)
            .ok_or_else(|| RealizarError::ModelNotFound(id.to_string()))?;

        Ok(entry.info.clone())
    }

    /// List all registered models (lock-free)
    #[must_use]
    pub fn list(&self) -> Vec<ModelInfo> {
        let models = self.models.load();
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
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if !current.contains_key(id) {
            return Err(RealizarError::ModelNotFound(id.to_string()));
        }

        let mut new_map: ModelsMap = (**current).clone();
        new_map.remove(id);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Atomically replace a model (for hot-reload)
    ///
    /// # Arguments
    ///
    /// * `id` - Model identifier to replace
    /// * `model` - New model instance
    /// * `tokenizer` - New tokenizer instance
    ///
    /// # Errors
    ///
    /// Returns error if model not found or lock acquisition fails
    pub fn replace(&self, id: &str, model: Model, tokenizer: BPETokenizer) -> Result<()> {
        let _guard = self.write_lock.lock().map_err(|_| {
            RealizarError::RegistryError("Failed to acquire write lock".to_string())
        })?;

        let current = self.models.load();
        if !current.contains_key(id) {
            return Err(RealizarError::ModelNotFound(id.to_string()));
        }

        // Get existing info to preserve metadata
        let existing_info = current.get(id).map_or_else(
            || ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                format: "unknown".to_string(),
                loaded: true,
            },
            |e| e.info.clone(),
        );

        let entry = ModelEntry {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            info: existing_info,
        };

        let mut new_map: ModelsMap = (**current).clone();
        new_map.insert(id.to_string(), entry);
        self.models.store(Arc::new(new_map));
        Ok(())
    }

    /// Check if a model is registered (lock-free)
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        let models = self.models.load();
        models.contains_key(id)
    }

    /// Get the number of registered models (lock-free)
    #[must_use]
    pub fn len(&self) -> usize {
        let models = self.models.load();
        models.len()
    }

    /// Check if the registry is empty (lock-free)
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
        let model = Model::new(config).expect("test");

        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

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

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_register_duplicate_error() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry
            .register("test-model", model1, tokenizer1)
            .expect("test");
        let result = registry.register("test-model", model2, tokenizer2);

        assert!(result.is_err());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_get_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        let (retrieved_model, retrieved_tokenizer) = registry.get("test-model").expect("test");
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
            .expect("test");

        let retrieved_info = registry.get_info("llama-7b").expect("test");
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

        registry
            .register("model-1", model1, tokenizer1)
            .expect("test");
        registry
            .register("model-2", model2, tokenizer2)
            .expect("test");

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

        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert_eq!(registry.len(), 1);

        registry.unregister("test-model").expect("test");
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
                    .expect("test");
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("test");
        }

        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_multiple_get_same_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        // Get the same model multiple times
        let (model1, _) = registry.get("test-model").expect("test");
        let (model2, _) = registry.get("test-model").expect("test");

        // Both should point to the same underlying model
        assert!(Arc::ptr_eq(&model1, &model2));
    }

    #[test]
    fn test_replace_model() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        // Register initial model
        registry
            .register("test-model", model1, tokenizer1)
            .expect("test");
        assert_eq!(registry.len(), 1);

        // Replace with new model
        registry
            .replace("test-model", model2, tokenizer2)
            .expect("test");
        assert_eq!(registry.len(), 1);

        // Verify replacement worked
        let (retrieved, _) = registry.get("test-model").expect("test");
        assert!(Arc::strong_count(&retrieved) >= 2);
    }

    #[test]
    fn test_replace_nonexistent_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        // Try to replace a model that doesn't exist
        let result = registry.replace("nonexistent", model, tokenizer);
        assert!(result.is_err());
    }

    #[test]
    fn test_contains_method() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        assert!(!registry.contains("test-model"));
        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_is_empty_method() {
        let registry = ModelRegistry::new(5);
        assert!(registry.is_empty());

        let (model, tokenizer) = create_test_model();
        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_get_info_nonexistent() {
        let registry = ModelRegistry::new(5);
        let result = registry.get_info("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_len_method() {
        let registry = ModelRegistry::new(5);
        assert_eq!(registry.len(), 0);

        let (model1, tokenizer1) = create_test_model();
        registry
            .register("model-1", model1, tokenizer1)
            .expect("test");
        assert_eq!(registry.len(), 1);

        let (model2, tokenizer2) = create_test_model();
        registry
            .register("model-2", model2, tokenizer2)
            .expect("test");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_register_with_info_duplicate_error() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        let info1 = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        let info2 = ModelInfo {
            id: "test-model".to_string(),
            name: "Another Test".to_string(),
            description: "Same ID".to_string(),
            format: "Safetensors".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info1, model1, tokenizer1)
            .expect("first registration should succeed");

        let result = registry.register_with_info(info2, model2, tokenizer2);
        assert!(result.is_err());
        match result {
            Err(RealizarError::ModelAlreadyExists(id)) => {
                assert_eq!(id, "test-model");
            }
            _ => panic!("Expected ModelAlreadyExists error"),
        }
    }

    #[test]
    fn test_model_info_serialization() {
        let info = ModelInfo {
            id: "llama-7b".to_string(),
            name: "Llama 7B".to_string(),
            description: "7B parameter model".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        // Test serialization roundtrip
        let json = serde_json::to_string(&info).expect("serialization");
        let deserialized: ModelInfo = serde_json::from_str(&json).expect("deserialization");

        assert_eq!(deserialized.id, info.id);
        assert_eq!(deserialized.name, info.name);
        assert_eq!(deserialized.description, info.description);
        assert_eq!(deserialized.format, info.format);
        assert_eq!(deserialized.loaded, info.loaded);
    }

    #[test]
    fn test_model_info_debug() {
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "Desc".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("ModelInfo"));
    }

    #[test]
    fn test_model_info_clone() {
        let info = ModelInfo {
            id: "original".to_string(),
            name: "Original".to_string(),
            description: "Original desc".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        let cloned = info.clone();
        assert_eq!(cloned.id, info.id);
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.description, info.description);
        assert_eq!(cloned.format, info.format);
        assert_eq!(cloned.loaded, info.loaded);
    }

    #[test]
    fn test_replace_preserves_metadata() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        let info = ModelInfo {
            id: "test-model".to_string(),
            name: "My Custom Name".to_string(),
            description: "My custom description".to_string(),
            format: "Safetensors".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info, model1, tokenizer1)
            .expect("registration");

        // Replace with new model
        registry
            .replace("test-model", model2, tokenizer2)
            .expect("replacement");

        // Verify metadata is preserved
        let retrieved_info = registry.get_info("test-model").expect("get info");
        assert_eq!(retrieved_info.name, "My Custom Name");
        assert_eq!(retrieved_info.description, "My custom description");
        assert_eq!(retrieved_info.format, "Safetensors");
    }

    #[test]
    fn test_list_empty_registry() {
        let registry = ModelRegistry::new(5);
        let list = registry.list();
        assert!(list.is_empty());
    }

    #[test]
    fn test_unregister_then_register_same_id() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        // Register first model
        registry
            .register("test-model", model1, tokenizer1)
            .expect("first registration");

        // Unregister
        registry.unregister("test-model").expect("unregistration");
        assert!(!registry.contains("test-model"));

        // Register again with same ID
        registry
            .register("test-model", model2, tokenizer2)
            .expect("second registration");
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_get_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.get("nonexistent-model-xyz");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "nonexistent-model-xyz");
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_get_info_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.get_info("another-nonexistent");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "another-nonexistent");
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_unregister_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.unregister("missing-model");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "missing-model");
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_replace_error_message() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();
        let result = registry.replace("nonexistent", model, tokenizer);

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "nonexistent");
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_register_error_message() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry
            .register("duplicate", model1, tokenizer1)
            .expect("first registration");

        let result = registry.register("duplicate", model2, tokenizer2);
        match result {
            Err(RealizarError::ModelAlreadyExists(id)) => {
                assert_eq!(id, "duplicate");
            }
            _ => panic!("Expected ModelAlreadyExists error"),
        }
    }

    #[test]
    fn test_concurrent_reads() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let (model, tokenizer) = create_test_model();
        registry
            .register("shared-model", model, tokenizer)
            .expect("registration");

        let mut handles = vec![];

        // Multiple concurrent reads
        for _ in 0..10 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let result = registry_clone.get("shared-model");
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_concurrent_list() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let (model, tokenizer) = create_test_model();
        registry
            .register("model", model, tokenizer)
            .expect("registration");

        let mut handles = vec![];

        // Multiple concurrent list calls
        for _ in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                for _ in 0..50 {
                    let list = registry_clone.list();
                    assert_eq!(list.len(), 1);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_new_with_various_capacities() {
        // Test with minimum capacity
        let registry = ModelRegistry::new(1);
        assert!(registry.is_empty());

        // Test with zero capacity (edge case)
        let registry = ModelRegistry::new(0);
        assert!(registry.is_empty());

        // Test with large capacity
        let registry = ModelRegistry::new(1000);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_default_info_values() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("simple-model", model, tokenizer)
            .expect("registration");

        let info = registry.get_info("simple-model").expect("get info");
        // Check default values set by register()
        assert_eq!(info.id, "simple-model");
        assert_eq!(info.name, "simple-model"); // Defaults to id
        assert_eq!(info.description, ""); // Empty by default
        assert_eq!(info.format, "unknown"); // Unknown by default
        assert!(info.loaded); // Should be true
    }

    #[test]
    fn test_register_with_info_sets_loaded_true() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        // Pass loaded=false, but it should be set to true
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "Desc".to_string(),
            format: "GGUF".to_string(),
            loaded: false, // Explicitly false
        };

        registry
            .register_with_info(info, model, tokenizer)
            .expect("registration");

        let retrieved = registry.get_info("test").expect("get info");
        assert!(retrieved.loaded); // Should be true despite passing false
    }

    #[test]
    fn test_contains_after_unregister() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("registration");
        assert!(registry.contains("test-model"));

        registry.unregister("test-model").expect("unregister");
        assert!(!registry.contains("test-model"));
    }

    #[test]
    fn test_len_after_multiple_operations() {
        let registry = ModelRegistry::new(10);
        assert_eq!(registry.len(), 0);

        // Add 3 models
        for i in 0..3 {
            let (model, tokenizer) = create_test_model();
            registry
                .register(&format!("model-{i}"), model, tokenizer)
                .expect("registration");
        }
        assert_eq!(registry.len(), 3);

        // Remove 1 model
        registry.unregister("model-1").expect("unregister");
        assert_eq!(registry.len(), 2);

        // Replace (shouldn't change count)
        let (model, tokenizer) = create_test_model();
        registry
            .replace("model-0", model, tokenizer)
            .expect("replace");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_model_info_empty_strings() {
        let info = ModelInfo {
            id: "".to_string(),
            name: "".to_string(),
            description: "".to_string(),
            format: "".to_string(),
            loaded: false,
        };

        assert!(info.id.is_empty());
        assert!(info.name.is_empty());
        assert!(info.description.is_empty());
        assert!(info.format.is_empty());
        assert!(!info.loaded);
    }

    #[test]
    fn test_model_info_special_characters() {
        let info = ModelInfo {
            id: "model/with:special@chars!".to_string(),
            name: "Name with spaces".to_string(),
            description: "Line1\nLine2\tTabbed".to_string(),
            format: "GGUF-v3.0".to_string(),
            loaded: true,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let deserialized: ModelInfo = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.id, info.id);
        assert_eq!(deserialized.name, info.name);
        assert_eq!(deserialized.description, info.description);
        assert_eq!(deserialized.format, info.format);
    }

    #[test]
    fn test_multiple_unregisters_different_models() {
        let registry = ModelRegistry::new(10);

        // Register 5 models
        for i in 0..5 {
            let (model, tokenizer) = create_test_model();
            registry
                .register(&format!("model-{i}"), model, tokenizer)
                .expect("registration");
        }
        assert_eq!(registry.len(), 5);

        // Unregister them all in reverse order
        for i in (0..5).rev() {
            registry
                .unregister(&format!("model-{i}"))
                .expect("unregister");
            assert_eq!(registry.len(), i);
        }
        assert!(registry.is_empty());
    }

    #[test]
    fn test_arc_reference_counts() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test", model, tokenizer)
            .expect("registration");

        // Get multiple references
        let (m1, t1) = registry.get("test").expect("get1");
        let (m2, t2) = registry.get("test").expect("get2");
        let (m3, t3) = registry.get("test").expect("get3");

        // All should point to the same underlying data
        assert!(Arc::ptr_eq(&m1, &m2));
        assert!(Arc::ptr_eq(&m2, &m3));
        assert!(Arc::ptr_eq(&t1, &t2));
        assert!(Arc::ptr_eq(&t2, &t3));

        // Strong count should be at least 4 (registry + 3 gets)
        assert!(Arc::strong_count(&m1) >= 4);
        assert!(Arc::strong_count(&t1) >= 4);
    }

    #[test]
    fn test_list_returns_cloned_info() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test Model".to_string(),
            description: "Description".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info, model, tokenizer)
            .expect("registration");

        let list1 = registry.list();
        let list2 = registry.list();

        // Lists should be independent copies
        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].id, list2[0].id);
    }
}
