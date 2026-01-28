//! Streaming KV Cache (PMAT-802)
//!
//! Memory-efficient key-value cache for transformer inference.

// ============================================================================
// M6: Memory Efficiency - StreamingKVCache
// ============================================================================

/// Streaming KV cache for memory-efficient inference
///
/// Implements a bounded circular buffer for key-value cache that allows
/// efficient inference on long sequences without unbounded memory growth.
///
/// ## Memory Bound
///
/// Total memory = num_layers * max_positions * num_heads * head_dim * 2 (K+V) * sizeof(f32)
///
/// For 7B model (32 layers, 2048 positions, 32 heads, 128 head_dim):
/// = 32 * 2048 * 32 * 128 * 2 * 4 = ~2GB
///
/// ## Usage
///
/// ```rust,ignore
/// let mut cache = StreamingKVCache::new(32, 2048, 32, 128);
/// cache.append(0, &key_vec, &value_vec);
/// let (keys, values) = cache.get_range(0, 0, 100);
/// ```
pub struct StreamingKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim]
    keys: Vec<Vec<f32>>,
    /// Value cache per layer
    values: Vec<Vec<f32>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCache {
    /// Create a new streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0.0f32; kv_size]; num_layers],
            values: vec![vec![0.0f32; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Append key-value pair for a single position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim]
    /// * `value` - Value vector [num_heads * head_dim]
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;
        self.keys[layer][offset..offset + kv_dim].copy_from_slice(key);
        self.values[layer][offset..offset + kv_dim].copy_from_slice(value);

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) slices
    #[must_use]
    pub fn get_range(&self, layer: usize, start: usize, end: usize) -> (&[f32], &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) for all valid positions
    #[must_use]
    pub fn get_valid(&self, layer: usize) -> (&[f32], &[f32]) {
        self.get_range(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
        // Note: We don't zero the memory for performance
    }

    /// Calculate memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, f32 = 4 bytes
        self.num_layers * kv_size * 2 * 4
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Streaming KV cache with FP16 storage for memory efficiency (M12)
///
/// Uses half-precision (FP16) storage to halve memory usage compared to FP32,
/// enabling support for ultra-long contexts (65536+) on consumer GPUs.
///
/// # Memory Efficiency
///
/// For 65536 context with 7B model config:
/// - FP32: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 4 bytes = 68.72 GB
/// - FP16: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 2 bytes = 34.36 GB
///
/// # Example
///
/// ```
/// use realizar::gpu::StreamingKVCacheFp16;
///
/// let mut cache = StreamingKVCacheFp16::new(32, 65536, 32, 128);
/// assert!(cache.memory_mb() < 36000.0); // < 36 GB
/// ```
pub struct StreamingKVCacheFp16 {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim] stored as FP16 bits
    keys: Vec<Vec<u16>>,
    /// Value cache per layer stored as FP16 bits
    values: Vec<Vec<u16>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCacheFp16 {
    /// Create a new FP16 streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0u16; kv_size]; num_layers],
            values: vec![vec![0u16; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Convert f32 to FP16 bits
    #[inline]
    pub(crate) fn f32_to_f16(value: f32) -> u16 {
        half::f16::from_f32(value).to_bits()
    }

    /// Convert FP16 bits to f32
    #[inline]
    pub(crate) fn f16_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// Append key-value pair for a single position (FP32 input, stored as FP16)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim] as FP32
    /// * `value` - Value vector [num_heads * head_dim] as FP32
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;

        // Convert FP32 to FP16 and store
        for (i, &k) in key.iter().enumerate() {
            self.keys[layer][offset + i] = Self::f32_to_f16(k);
        }
        for (i, &v) in value.iter().enumerate() {
            self.values[layer][offset + i] = Self::f32_to_f16(v);
        }

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions (converted back to FP32)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) as Vec<f32>
    #[must_use]
    pub fn get_range_f32(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        let keys: Vec<f32> = self.keys[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        let values: Vec<f32> = self.values[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        (keys, values)
    }

    /// Get raw FP16 keys and values for a range of positions
    #[must_use]
    pub fn get_range_raw(&self, layer: usize, start: usize, end: usize) -> (&[u16], &[u16]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer (as FP32)
    #[must_use]
    pub fn get_valid_f32(&self, layer: usize) -> (Vec<f32>, Vec<f32>) {
        self.get_range_f32(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
    }

    /// Calculate memory usage in bytes (half of FP32 version)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, u16 (FP16) = 2 bytes
        self.num_layers * kv_size * 2 * 2
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StreamingKVCache Tests
    // =========================================================================

    #[test]
    fn test_streaming_kv_cache_new() {
        let cache = StreamingKVCache::new(4, 128, 8, 64);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_positions(), 128);
    }

    #[test]
    fn test_streaming_kv_cache_append() {
        let mut cache = StreamingKVCache::new(2, 16, 4, 32);
        let kv_dim = 4 * 32;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(1, &key, &value);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_append_multiple() {
        let mut cache = StreamingKVCache::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key1 = vec![1.0f32; kv_dim];
        let value1 = vec![2.0f32; kv_dim];
        let key2 = vec![3.0f32; kv_dim];
        let value2 = vec![4.0f32; kv_dim];

        // First position
        cache.append(0, &key1, &value1);
        cache.append(1, &key1, &value1);
        assert_eq!(cache.len(), 1);

        // Second position
        cache.append(0, &key2, &value2);
        cache.append(1, &key2, &value2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_streaming_kv_cache_get_range() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.5f32; kv_dim];
        let value = vec![2.5f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys, values) = cache.get_range(0, 0, 1);
        assert_eq!(keys.len(), kv_dim);
        assert_eq!(values.len(), kv_dim);
        assert!((keys[0] - 1.5).abs() < 0.01);
        assert!((values[0] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_get_valid() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(0, &key, &value);

        let (keys, values) = cache.get_valid(0);
        assert_eq!(keys.len(), 2 * kv_dim);
        assert_eq!(values.len(), 2 * kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_circular_buffer() {
        // Test that cache wraps around when full
        let mut cache = StreamingKVCache::new(1, 4, 1, 2);
        let kv_dim = 1 * 2;

        // Fill cache
        for i in 0..4 {
            let key = vec![i as f32; kv_dim];
            let value = vec![(i * 10) as f32; kv_dim];
            cache.append(0, &key, &value);
        }
        assert_eq!(cache.len(), 4);

        // Overflow: should wrap around
        let key = vec![100.0f32; kv_dim];
        let value = vec![200.0f32; kv_dim];
        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 4); // Still 4, max_positions

        // First position should now have the overwritten value
        let (keys, _) = cache.get_range(0, 0, 1);
        assert!((keys[0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_clear() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_memory_bytes() {
        // 2 layers, 128 positions, 4 heads, 32 dim
        // kv_size = 128 * 4 * 32 = 16384
        // memory = 2 * 16384 * 2 * 4 = 262144 bytes = 256 KB
        let cache = StreamingKVCache::new(2, 128, 4, 32);
        assert_eq!(cache.memory_bytes(), 262_144);
    }

    #[test]
    fn test_streaming_kv_cache_memory_mb() {
        let cache = StreamingKVCache::new(2, 128, 4, 32);
        let mb = cache.memory_mb();
        assert!((mb - 0.25).abs() < 0.01); // 256 KB = 0.25 MB
    }

    #[test]
    #[should_panic(expected = "Layer index out of bounds")]
    fn test_streaming_kv_cache_invalid_layer() {
        let mut cache = StreamingKVCache::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];
        cache.append(2, &key, &value); // layer 2 is out of bounds (0, 1)
    }

    #[test]
    #[should_panic(expected = "Key dimension mismatch")]
    fn test_streaming_kv_cache_key_dimension_mismatch() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let key = vec![1.0f32; 4]; // Wrong dimension (should be 8)
        let value = vec![2.0f32; 8];
        cache.append(0, &key, &value);
    }

    // =========================================================================
    // StreamingKVCacheFp16 Tests
    // =========================================================================

    #[test]
    fn test_streaming_kv_cache_fp16_new() {
        let cache = StreamingKVCacheFp16::new(4, 128, 8, 64);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_positions(), 128);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_conversion() {
        // Test f32 -> f16 -> f32 round trip
        let val = 1.5f32;
        let fp16 = StreamingKVCacheFp16::f32_to_f16(val);
        let back = StreamingKVCacheFp16::f16_to_f32(fp16);
        assert!((back - val).abs() < 0.001);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_append() {
        let mut cache = StreamingKVCacheFp16::new(2, 16, 4, 32);
        let kv_dim = 4 * 32;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(1, &key, &value);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_range_f32() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.5f32; kv_dim];
        let value = vec![2.5f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys, values) = cache.get_range_f32(0, 0, 1);
        assert_eq!(keys.len(), kv_dim);
        assert_eq!(values.len(), kv_dim);
        // FP16 has some loss of precision
        assert!((keys[0] - 1.5).abs() < 0.01);
        assert!((values[0] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_range_raw() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys_raw, values_raw) = cache.get_range_raw(0, 0, 1);
        assert_eq!(keys_raw.len(), kv_dim);
        assert_eq!(values_raw.len(), kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_valid_f32() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(0, &key, &value);

        let (keys, values) = cache.get_valid_f32(0);
        assert_eq!(keys.len(), 2 * kv_dim);
        assert_eq!(values.len(), 2 * kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_clear() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_fp16_memory_bytes() {
        // 2 layers, 128 positions, 4 heads, 32 dim
        // kv_size = 128 * 4 * 32 = 16384
        // memory = 2 * 16384 * 2 * 2 = 131072 bytes = 128 KB
        // (half of FP32 version)
        let cache = StreamingKVCacheFp16::new(2, 128, 4, 32);
        assert_eq!(cache.memory_bytes(), 131_072);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_memory_mb() {
        let cache = StreamingKVCacheFp16::new(2, 128, 4, 32);
        let mb = cache.memory_mb();
        assert!((mb - 0.125).abs() < 0.01); // 128 KB = 0.125 MB
    }

    #[test]
    fn test_streaming_kv_cache_fp16_half_memory_of_fp32() {
        let fp32_cache = StreamingKVCache::new(4, 256, 8, 64);
        let fp16_cache = StreamingKVCacheFp16::new(4, 256, 8, 64);

        // FP16 should use exactly half the memory of FP32
        assert_eq!(fp16_cache.memory_bytes(), fp32_cache.memory_bytes() / 2);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_circular_buffer() {
        let mut cache = StreamingKVCacheFp16::new(1, 4, 1, 2);
        let kv_dim = 1 * 2;

        // Fill cache
        for i in 0..4 {
            let key = vec![i as f32; kv_dim];
            let value = vec![(i * 10) as f32; kv_dim];
            cache.append(0, &key, &value);
        }
        assert_eq!(cache.len(), 4);

        // Overflow: should wrap around
        let key = vec![100.0f32; kv_dim];
        let value = vec![200.0f32; kv_dim];
        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 4);

        // First position should now have the overwritten value
        let (keys, _) = cache.get_range_f32(0, 0, 1);
        assert!((keys[0] - 100.0).abs() < 0.1); // FP16 precision
    }

    #[test]
    #[should_panic(expected = "Layer index out of bounds")]
    fn test_streaming_kv_cache_fp16_invalid_layer() {
        let mut cache = StreamingKVCacheFp16::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];
        cache.append(2, &key, &value);
    }

    #[test]
    #[should_panic(expected = "Value dimension mismatch")]
    fn test_streaming_kv_cache_fp16_value_dimension_mismatch() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let key = vec![1.0f32; 8];
        let value = vec![2.0f32; 4]; // Wrong dimension (should be 8)
        cache.append(0, &key, &value);
    }
}
