//! Model caching and warming for reduced latency
//!
//! Provides LRU cache for model instances to reduce cold start latency and
//! improve throughput for repeated model usage.
//!
//! ## Features
//!
//! - **LRU Eviction**: Least Recently Used models are evicted when cache is full
//! - **Cache Warming**: Pre-load models on startup for zero cold starts
//! - **Metrics**: Track cache hits, misses, and evictions
//! - **Thread-Safe**: Concurrent access via Arc<RwLock>
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::cache::ModelCache;
//!
//! let cache = ModelCache::new(10); // capacity: 10 models
//! cache.warm(&["model1", "model2"])?;
//!
//! let model = cache.get_or_load("model1", || load_model("model1"))?;
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::{
    error::RealizarError,
    layers::{Model, ModelConfig},
    tokenizer::BPETokenizer,
};

/// Type alias for model and tokenizer pair
pub type ModelPair = (Arc<Model>, Arc<BPETokenizer>);

/// Type alias for the internal cache storage
type CacheStorage = Arc<RwLock<HashMap<CacheKey, CacheEntry>>>;

/// Cache key for identifying cached models
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Model identifier (name, path, or config hash)
    pub id: String,
}

impl CacheKey {
    /// Create a new cache key
    #[must_use]
    pub fn new(id: String) -> Self {
        Self { id }
    }

    /// Create cache key from model config
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        // Simple hash based on config parameters
        let id = format!(
            "v{}_h{}_n{}_l{}_i{}",
            config.vocab_size,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
            config.intermediate_dim
        );
        Self::new(id)
    }
}

/// Cached model entry with metadata
#[derive(Clone)]
pub struct CacheEntry {
    /// The cached model
    pub model: Arc<Model>,
    /// The tokenizer for this model
    pub tokenizer: Arc<BPETokenizer>,
    /// Access count for LRU tracking
    access_count: u64,
    /// Last access timestamp
    last_access: std::time::Instant,
}

impl CacheEntry {
    /// Create a new cache entry
    #[must_use]
    pub fn new(model: Model, tokenizer: BPETokenizer) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            access_count: 0,
            last_access: std::time::Instant::now(),
        }
    }

    /// Record an access to this entry
    fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = std::time::Instant::now();
    }
}

/// Cache metrics for monitoring
#[derive(Debug, Default, Clone)]
pub struct CacheMetrics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current cache size
    pub size: usize,
}

impl CacheMetrics {
    /// Calculate hit rate as percentage
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// LRU model cache
pub struct ModelCache {
    /// Cached entries
    cache: CacheStorage,
    /// Maximum cache capacity
    capacity: usize,
    /// Cache metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

impl ModelCache {
    /// Create a new model cache with the specified capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of models to cache
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            capacity,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Get a model from cache or load it using the provided function
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key for the model
    /// * `loader` - Function to load the model if not cached
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned (extremely rare, indicates thread panic while holding lock)
    pub fn get_or_load<F>(&self, key: &CacheKey, loader: F) -> Result<ModelPair, RealizarError>
    where
        F: FnOnce() -> Result<(Model, BPETokenizer), RealizarError>,
    {
        // Try to get from cache first (read lock)
        {
            let mut cache = self
                .cache
                .write()
                .expect("RwLock poisoned: thread panicked while holding cache write lock");
            if let Some(entry) = cache.get_mut(key) {
                entry.record_access();
                let mut metrics = self
                    .metrics
                    .write()
                    .expect("RwLock poisoned: thread panicked while holding metrics write lock");
                metrics.hits += 1;
                return Ok((entry.model.clone(), entry.tokenizer.clone()));
            }
        }

        // Cache miss - load the model
        let (model, tokenizer) = loader()?;
        let entry = CacheEntry::new(model, tokenizer);

        // Insert into cache (write lock)
        {
            let mut cache = self
                .cache
                .write()
                .expect("RwLock poisoned: thread panicked while holding cache write lock");
            let mut metrics = self
                .metrics
                .write()
                .expect("RwLock poisoned: thread panicked while holding metrics write lock");

            metrics.misses += 1;

            // Check if we need to evict
            if cache.len() >= self.capacity && !cache.contains_key(key) {
                Self::evict_lru(&mut cache, &mut metrics);
            }

            cache.insert(key.clone(), entry.clone());
            metrics.size = cache.len();
        }

        Ok((entry.model, entry.tokenizer))
    }

    /// Evict the least recently used entry from the cache
    fn evict_lru(cache: &mut HashMap<CacheKey, CacheEntry>, metrics: &mut CacheMetrics) {
        if let Some((lru_key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, e)| (k.clone(), e.clone()))
        {
            cache.remove(&lru_key);
            metrics.evictions += 1;
        }
    }

    /// Get current cache metrics
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn metrics(&self) -> CacheMetrics {
        self.metrics
            .read()
            .expect("RwLock poisoned: thread panicked while holding metrics read lock")
            .clone()
    }

    /// Clear all cached models
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    pub fn clear(&self) {
        let mut cache = self
            .cache
            .write()
            .expect("RwLock poisoned: thread panicked while holding cache write lock");
        cache.clear();
        let mut metrics = self
            .metrics
            .write()
            .expect("RwLock poisoned: thread panicked while holding metrics write lock");
        metrics.size = 0;
    }

    /// Get the number of cached models
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache
            .read()
            .expect("RwLock poisoned: thread panicked while holding cache read lock")
            .len()
    }

    /// Check if the cache is empty
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestModelResult = Result<(Model, BPETokenizer), RealizarError>;

    fn create_test_model(vocab_size: usize) -> TestModelResult {
        let config = ModelConfig {
            vocab_size,
            hidden_dim: 16,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 32,
            eps: 1e-5,
        };
        let model = Model::new(config)?;

        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        Ok((model, tokenizer))
    }

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new("model1".to_string());
        assert_eq!(key.id, "model1");
    }

    #[test]
    fn test_cache_key_from_config() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 2,
            num_layers: 4,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let key = CacheKey::from_config(&config);
        assert_eq!(key.id, "v100_h32_n2_l4_i64");
    }

    #[test]
    fn test_cache_creation() {
        let cache = ModelCache::new(10);
        assert_eq!(cache.capacity, 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit() {
        let cache = ModelCache::new(10);
        let key = CacheKey::new("test".to_string());

        // First access - cache miss
        let result1 = cache.get_or_load(&key, || create_test_model(50));
        assert!(result1.is_ok());

        let metrics1 = cache.metrics();
        assert_eq!(metrics1.hits, 0);
        assert_eq!(metrics1.misses, 1);
        assert_eq!(metrics1.size, 1);

        // Second access - cache hit
        let result2 = cache.get_or_load(&key, || create_test_model(50));
        assert!(result2.is_ok());

        let metrics2 = cache.metrics();
        assert_eq!(metrics2.hits, 1);
        assert_eq!(metrics2.misses, 1);
        assert_eq!(metrics2.size, 1);
    }

    #[test]
    fn test_cache_miss() {
        let cache = ModelCache::new(10);
        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());

        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 2);
        assert_eq!(metrics.size, 2);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ModelCache::new(2); // Small capacity

        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());
        let key3 = CacheKey::new("model3".to_string());

        // Fill cache to capacity
        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        assert_eq!(cache.len(), 2);

        // Add third model - should evict LRU (model1)
        cache
            .get_or_load(&key3, || create_test_model(70))
            .expect("test");

        assert_eq!(cache.len(), 2);
        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 1);
    }

    #[test]
    fn test_cache_clear() {
        let cache = ModelCache::new(10);

        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());

        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut metrics = CacheMetrics::default();
        assert!(metrics.hit_rate().abs() < 0.1);

        metrics.hits = 7;
        metrics.misses = 3;
        assert!((metrics.hit_rate() - 70.0).abs() < 0.1);

        metrics.hits = 100;
        metrics.misses = 0;
        assert!((metrics.hit_rate() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_cache_entry_access_tracking() {
        let (model, tokenizer) = create_test_model(50).expect("test");
        let mut entry = CacheEntry::new(model, tokenizer);

        assert_eq!(entry.access_count, 0);

        entry.record_access();
        assert_eq!(entry.access_count, 1);

        entry.record_access();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let cache = Arc::new(ModelCache::new(10));
        let key = CacheKey::new("concurrent".to_string());

        // Pre-populate cache to avoid race condition on first load
        cache
            .get_or_load(&key, || create_test_model(50))
            .expect("test");

        // Reset metrics to get clean counts
        let initial_metrics = cache.metrics();
        let initial_hits = initial_metrics.hits;
        let initial_misses = initial_metrics.misses;

        // Spawn multiple threads accessing the same key (all should hit)
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let cache = cache.clone();
                let key = key.clone();
                thread::spawn(move || {
                    cache
                        .get_or_load(&key, || create_test_model(50))
                        .expect("test");
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("test");
        }

        // All threads should have hit the cache
        let metrics = cache.metrics();
        assert_eq!(metrics.size, 1);
        assert_eq!(metrics.hits, initial_hits + 5); // Exactly 5 hits
        assert_eq!(metrics.misses, initial_misses); // No new misses
    }
}
