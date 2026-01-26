//! Dequantized weight cache for GPU GEMM operations
//!
//! Stores pre-dequantized f32 weights for GPU GEMM to avoid
//! repeated dequantization on every forward pass.

/// Dequantized FFN weights for a single transformer layer
///
/// Stores pre-dequantized f32 weights for GPU GEMM operations.
/// Cache these to avoid repeated dequantization on every forward pass.
#[derive(Clone)]
pub struct DequantizedFFNWeights {
    /// Up projection weights [hidden_dim, intermediate_dim]
    pub up: Vec<f32>,
    /// Down projection weights [intermediate_dim, hidden_dim]
    pub down: Vec<f32>,
    /// Optional up bias [intermediate_dim]
    pub up_bias: Option<Vec<f32>>,
    /// Optional down bias [hidden_dim]
    pub down_bias: Option<Vec<f32>>,
}

/// Cache for dequantized FFN weights (PARITY-019)
///
/// Uses RwLock for concurrent read access during batch inference.
/// Weights are dequantized once during warmup and reused for GPU GEMM.
///
/// # Performance Impact
/// - Eliminates per-forward dequantization overhead
/// - Enables GPU GEMM with f32 weights
/// - Memory tradeoff: ~6.4 GB for phi-2 32 layers
///
/// # Thread Safety
/// - RwLock allows multiple concurrent readers during inference
/// - Single writer during warmup phase
#[cfg(feature = "gpu")]
pub struct DequantizedWeightCache {
    /// Per-layer dequantized weights
    layers: std::sync::RwLock<std::collections::HashMap<usize, DequantizedFFNWeights>>,
    /// Hidden dimension for validation
    hidden_dim: usize,
    /// Intermediate FFN dimension
    intermediate_dim: usize,
    /// Number of layers to cache
    num_layers: usize,
}

#[cfg(feature = "gpu")]
impl DequantizedWeightCache {
    /// Create a new weight cache with specified dimensions
    ///
    /// # Arguments
    /// * `hidden_dim` - Model hidden dimension (e.g., 2560 for phi-2)
    /// * `intermediate_dim` - FFN intermediate dimension (e.g., 10240 for phi-2)
    /// * `num_layers` - Number of transformer layers to cache
    #[must_use]
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_layers: usize) -> Self {
        Self {
            layers: std::sync::RwLock::new(std::collections::HashMap::with_capacity(num_layers)),
            hidden_dim,
            intermediate_dim,
            num_layers,
        }
    }

    /// Pre-warmup all layers with dequantized weights
    ///
    /// Call this once at startup to avoid dequantization during inference.
    /// The closure receives layer index and returns (up_weights, down_weights).
    ///
    /// # Arguments
    /// * `dequant_fn` - Closure that dequantizes weights for a given layer index
    ///
    /// # Panics
    /// Panics if the RwLock is poisoned
    pub fn warmup<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias: None,
                    down_bias: None,
                }
            });
        }
    }

    /// Warmup with biases
    ///
    /// Same as `warmup` but also caches bias vectors.
    pub fn warmup_with_bias<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down, up_bias, down_bias) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias,
                    down_bias,
                }
            });
        }
    }

    /// Get cached weights for a layer (read-only access)
    ///
    /// Returns None if the layer hasn't been warmed up.
    /// Uses read lock for concurrent access during batch inference.
    pub fn get(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.get(&layer_idx).cloned()
    }

    /// Check if a layer is cached
    pub fn is_cached(&self, layer_idx: usize) -> bool {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.contains_key(&layer_idx)
    }

    /// Get number of cached layers
    pub fn cached_count(&self) -> usize {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.len()
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Each layer: up + down weights
        // up: hidden_dim × intermediate_dim × 4 bytes
        // down: intermediate_dim × hidden_dim × 4 bytes
        let per_layer = 2 * self.hidden_dim * self.intermediate_dim * 4;
        self.cached_count() * per_layer
    }

    /// Get model dimensions
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.hidden_dim, self.intermediate_dim, self.num_layers)
    }
}
