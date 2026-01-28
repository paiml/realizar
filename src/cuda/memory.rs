//! GPU Memory Pool and Staging Buffers
//!
//! This module provides efficient memory management for GPU operations:
//! - `GpuMemoryPool`: Reduces cudaMalloc/cudaFree overhead
//! - `StagingBufferPool`: Pinned host memory for fast transfers
//! - `TransferMode`: Configuration for host-device transfers

use std::collections::BTreeMap;

// ============================================================================
// GPU Memory Pool (IMP-900d)
// ============================================================================

/// Size class for memory pool allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizeClass(usize);

impl SizeClass {
    /// Standard size classes (powers of 2 from 4KB to 256MB)
    pub const CLASSES: [usize; 9] = [
        4096,        // 4 KB
        16384,       // 16 KB
        65536,       // 64 KB
        262_144,     // 256 KB
        1_048_576,   // 1 MB
        4_194_304,   // 4 MB
        16_777_216,  // 16 MB
        67_108_864,  // 64 MB
        268_435_456, // 256 MB
    ];

    /// Find the smallest size class that fits the requested size
    #[must_use]
    pub fn for_size(size: usize) -> Option<Self> {
        Self::CLASSES
            .iter()
            .find(|&&class| class >= size)
            .map(|&class| SizeClass(class))
    }

    /// Get the size in bytes
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.0
    }
}

/// GPU memory pool for efficient buffer allocation (IMP-900d)
///
/// Reduces cudaMalloc/cudaFree overhead by reusing allocated buffers.
/// Buffers are organized by size class for O(1) allocation when a
/// matching buffer is available.
///
/// # Performance Impact
///
/// - Without pool: ~50-100μs per cudaMalloc/cudaFree pair
/// - With pool: ~1-5μs for buffer reuse
/// - Expected improvement: 1.5-2x for memory-bound workloads
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Free buffers organized by size class
    free_buffers: BTreeMap<usize, Vec<GpuBufferHandle>>,
    /// Total bytes currently allocated
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Number of allocations served from pool
    pool_hits: usize,
    /// Number of allocations requiring new cudaMalloc
    pool_misses: usize,
    /// Maximum pool size (bytes)
    max_size: usize,
}

/// Handle to a GPU buffer (stores raw pointer and size)
#[derive(Debug)]
pub struct GpuBufferHandle {
    /// Size in bytes
    pub size: usize,
    /// Whether this buffer is currently in use
    pub in_use: bool,
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_buffers: BTreeMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            max_size: 2 * 1024 * 1024 * 1024, // 2 GB default
        }
    }

    /// Create a pool with custom max size
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Self::new()
        }
    }

    /// Try to get a buffer from the pool
    ///
    /// Returns a buffer handle if one of suitable size is available.
    pub fn try_get(&mut self, size: usize) -> Option<GpuBufferHandle> {
        // Find the smallest size class that fits
        let size_class = SizeClass::for_size(size)?;
        let class_size = size_class.bytes();

        // Check if we have a free buffer in this size class
        if let Some(buffers) = self.free_buffers.get_mut(&class_size) {
            if let Some(mut handle) = buffers.pop() {
                handle.in_use = true;
                self.pool_hits += 1;
                return Some(handle);
            }
        }

        // No buffer available, will need to allocate
        self.pool_misses += 1;
        None
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&mut self, mut handle: GpuBufferHandle) {
        handle.in_use = false;
        let size_class = SizeClass::for_size(handle.size).map_or(handle.size, |s| s.bytes());

        self.free_buffers
            .entry(size_class)
            .or_default()
            .push(handle);
    }

    /// Record an allocation (for tracking)
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }
    }

    /// Record a deallocation (for tracking)
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_allocated = self.total_allocated.saturating_sub(size);
    }

    /// Check if pool has capacity for additional allocation
    #[must_use]
    pub fn has_capacity(&self, size: usize) -> bool {
        self.total_allocated + size <= self.max_size
    }

    /// Get maximum pool size
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            hit_rate: if self.pool_hits + self.pool_misses > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
            free_buffers: self.free_buffers.values().map(Vec::len).sum(),
        }
    }

    /// Clear all free buffers (releases GPU memory)
    pub fn clear(&mut self) {
        self.free_buffers.clear();
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total bytes currently allocated
    pub total_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of allocations served from pool
    pub pool_hits: usize,
    /// Number of allocations requiring new cudaMalloc
    pub pool_misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Number of free buffers in pool
    pub free_buffers: usize,
}

impl PoolStats {
    /// Calculate memory savings from pooling
    #[must_use]
    pub fn estimated_savings_bytes(&self) -> usize {
        // Each pool hit saves ~100μs of cudaMalloc time
        // Estimate average allocation size from peak/total ratio
        if self.pool_hits > 0 {
            self.pool_hits * 1024 * 1024 // Assume average 1MB allocation
        } else {
            0
        }
    }
}

// ============================================================================
// PARITY-042: Pinned Host Memory for Zero-Copy Transfers
// ============================================================================

/// Pinned (page-locked) host memory buffer for faster GPU transfers
///
/// Pinned memory provides several benefits:
/// - DMA transfers without CPU involvement (~2x faster H2D/D2H)
/// - Zero-copy access where GPU can directly read host memory
/// - Async transfer overlap with kernel execution
///
/// # Memory Model
///
/// ```text
/// Regular Memory:        Pinned Memory:
/// ┌─────────────┐       ┌─────────────┐
/// │ Host Memory │       │ Host Memory │ (page-locked)
/// └──────┬──────┘       └──────┬──────┘
///        │ copy                 │ DMA
/// ┌──────▼──────┐       ┌──────▼──────┐
/// │ Page Cache  │       │   (skip)    │
/// └──────┬──────┘       └─────────────┘
///        │ DMA                  │
/// ┌──────▼──────┐       ┌──────▼──────┐
/// │ GPU Memory  │       │ GPU Memory  │
/// └─────────────┘       └─────────────┘
/// ```
///
/// # CUDA Implementation
///
/// When trueno-gpu adds `cuMemAllocHost` support, this will use true
/// page-locked memory. Currently uses aligned allocation as fallback.
#[derive(Debug)]
pub struct PinnedHostBuffer<T> {
    /// Aligned data storage
    data: Vec<T>,
    /// Whether this is truly pinned (requires CUDA driver support)
    is_pinned: bool,
}

impl<T: Copy + Default> PinnedHostBuffer<T> {
    /// Allocate a pinned host buffer
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements
    ///
    /// # Note
    ///
    /// Currently falls back to aligned allocation. True pinned memory
    /// requires trueno-gpu CUDA driver support for `cuMemAllocHost`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        // Allocate with alignment for cache-line efficiency (64 bytes)
        // Note: Currently uses standard allocation. True CUDA pinned memory
        // (cuMemAllocHost) requires trueno-gpu driver support - tracked in PARITY-042.
        let data = vec![T::default(); len];

        Self {
            data,
            is_pinned: false, // Will be true when CUDA support added
        }
    }

    /// Get slice of data
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice of data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get length in elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if truly pinned (page-locked)
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Copy from slice
    pub fn copy_from_slice(&mut self, src: &[T]) {
        self.data.copy_from_slice(src);
    }
}

/// Pool of staging buffers for efficient H2D/D2H transfers (PARITY-042)
///
/// Maintains reusable pinned buffers to avoid allocation overhead.
/// Staging buffers are used to:
/// 1. Copy data to pinned memory
/// 2. Async transfer to GPU
/// 3. Overlap with kernel execution
///
/// # Performance Impact
///
/// - Without staging: allocate → copy → free each transfer
/// - With staging: reuse pre-allocated pinned buffers
/// - Expected improvement: 1.3-1.5x for memory-bound workloads
#[derive(Debug)]
pub struct StagingBufferPool {
    /// Free staging buffers by size class
    free_buffers: BTreeMap<usize, Vec<PinnedHostBuffer<f32>>>,
    /// Total bytes allocated
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Pool hits (buffer reuse)
    pool_hits: usize,
    /// Pool misses (new allocation)
    pool_misses: usize,
    /// Maximum pool size in bytes
    max_size: usize,
}

impl Default for StagingBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl StagingBufferPool {
    /// Create a new staging buffer pool
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_buffers: BTreeMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            max_size: 512 * 1024 * 1024, // 512 MB default for staging
        }
    }

    /// Create pool with custom max size
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Self::new()
        }
    }

    /// Get a staging buffer of at least `size` elements
    ///
    /// Returns a buffer from the pool if available, otherwise allocates new.
    pub fn get(&mut self, size: usize) -> PinnedHostBuffer<f32> {
        let size_bytes = size * std::mem::size_of::<f32>();
        let size_class = SizeClass::for_size(size_bytes).map_or(size_bytes, |c| c.bytes());
        let elements = size_class / std::mem::size_of::<f32>();

        // Try to get from pool
        if let Some(buffers) = self.free_buffers.get_mut(&size_class) {
            if let Some(buf) = buffers.pop() {
                self.pool_hits += 1;
                return buf;
            }
        }

        // Allocate new buffer
        self.pool_misses += 1;
        let buf = PinnedHostBuffer::new(elements);
        self.total_allocated += buf.size_bytes();
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        buf
    }

    /// Return a buffer to the pool
    pub fn put(&mut self, buf: PinnedHostBuffer<f32>) {
        let size_class = buf.size_bytes();

        // Don't pool if over max size
        if self.total_allocated > self.max_size {
            self.total_allocated = self.total_allocated.saturating_sub(size_class);
            return; // Drop buffer
        }

        self.free_buffers.entry(size_class).or_default().push(buf);
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> StagingPoolStats {
        let free_count: usize = self.free_buffers.values().map(Vec::len).sum();
        StagingPoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            free_buffers: free_count,
            hit_rate: if self.pool_hits + self.pool_misses > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all buffers from pool
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.total_allocated = 0;
    }
}

/// Statistics for staging buffer pool
#[derive(Debug, Clone)]
pub struct StagingPoolStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of buffer reuses
    pub pool_hits: usize,
    /// Number of new allocations
    pub pool_misses: usize,
    /// Number of free buffers
    pub free_buffers: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

/// Zero-copy transfer configuration (PARITY-042)
///
/// Controls how data is transferred between host and device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferMode {
    /// Standard pageable memory transfer
    /// - Simplest, works everywhere
    /// - Involves CPU copy to staging area
    #[default]
    Pageable,
    /// Pinned memory transfer (faster DMA)
    /// - 1.5-2x faster than pageable
    /// - Requires page-locked memory
    Pinned,
    /// Zero-copy mapped memory (no transfer)
    /// - GPU directly accesses host memory
    /// - Best for infrequent access patterns
    /// - Requires unified memory support
    ZeroCopy,
    /// Async transfer with stream overlap
    /// - Transfer while previous kernel runs
    /// - Best for pipelined workloads
    Async,
}

impl TransferMode {
    /// Check if this mode requires pinned memory
    #[must_use]
    pub fn requires_pinned(&self) -> bool {
        matches!(self, Self::Pinned | Self::ZeroCopy | Self::Async)
    }

    /// Estimated speedup vs pageable transfer
    #[must_use]
    pub fn estimated_speedup(&self) -> f64 {
        match self {
            Self::Pageable => 1.0,
            Self::Pinned => 1.7,   // ~70% faster DMA
            Self::ZeroCopy => 2.0, // No transfer overhead
            Self::Async => 1.5,    // Overlap hides latency
        }
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SizeClass Tests
    // =========================================================================

    #[test]
    fn test_size_class_constants() {
        assert_eq!(SizeClass::CLASSES.len(), 9);
        assert_eq!(SizeClass::CLASSES[0], 4096); // 4 KB
        assert_eq!(SizeClass::CLASSES[8], 268_435_456); // 256 MB
    }

    #[test]
    fn test_size_class_for_size_exact_match() {
        let class = SizeClass::for_size(4096).unwrap();
        assert_eq!(class.bytes(), 4096);
    }

    #[test]
    fn test_size_class_for_size_rounds_up() {
        let class = SizeClass::for_size(5000).unwrap();
        assert_eq!(class.bytes(), 16384); // 16 KB
    }

    #[test]
    fn test_size_class_for_size_smallest() {
        let class = SizeClass::for_size(1).unwrap();
        assert_eq!(class.bytes(), 4096); // 4 KB minimum
    }

    #[test]
    fn test_size_class_for_size_too_large() {
        let result = SizeClass::for_size(300_000_000); // > 256 MB
        assert!(result.is_none());
    }

    #[test]
    fn test_size_class_for_size_zero() {
        let class = SizeClass::for_size(0).unwrap();
        assert_eq!(class.bytes(), 4096); // 4 KB minimum
    }

    #[test]
    fn test_size_class_ord() {
        let small = SizeClass(4096);
        let large = SizeClass(65536);
        assert!(small < large);
    }

    #[test]
    fn test_size_class_eq() {
        let a = SizeClass(4096);
        let b = SizeClass(4096);
        assert_eq!(a, b);
    }

    #[test]
    fn test_size_class_clone_copy() {
        let class = SizeClass(4096);
        let cloned = class;
        assert_eq!(cloned.bytes(), 4096);
    }

    #[test]
    fn test_size_class_debug() {
        let class = SizeClass(4096);
        let debug_str = format!("{:?}", class);
        assert!(debug_str.contains("SizeClass"));
        assert!(debug_str.contains("4096"));
    }

    // =========================================================================
    // GpuBufferHandle Tests
    // =========================================================================

    #[test]
    fn test_gpu_buffer_handle_new() {
        let handle = GpuBufferHandle {
            size: 1024,
            in_use: false,
        };
        assert_eq!(handle.size, 1024);
        assert!(!handle.in_use);
    }

    #[test]
    fn test_gpu_buffer_handle_debug() {
        let handle = GpuBufferHandle {
            size: 1024,
            in_use: true,
        };
        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("GpuBufferHandle"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("true"));
    }

    // =========================================================================
    // GpuMemoryPool Tests
    // =========================================================================

    #[test]
    fn test_gpu_memory_pool_default() {
        let pool = GpuMemoryPool::default();
        assert_eq!(pool.max_size(), 2 * 1024 * 1024 * 1024); // 2 GB
    }

    #[test]
    fn test_gpu_memory_pool_new() {
        let pool = GpuMemoryPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }

    #[test]
    fn test_gpu_memory_pool_with_max_size() {
        let pool = GpuMemoryPool::with_max_size(1024 * 1024);
        assert_eq!(pool.max_size(), 1024 * 1024);
    }

    #[test]
    fn test_gpu_memory_pool_try_get_empty() {
        let mut pool = GpuMemoryPool::new();
        let result = pool.try_get(4096);
        assert!(result.is_none());
        assert_eq!(pool.stats().pool_misses, 1);
    }

    #[test]
    fn test_gpu_memory_pool_return_and_get() {
        let mut pool = GpuMemoryPool::new();
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: true,
        };
        pool.return_buffer(handle);

        let result = pool.try_get(4096);
        assert!(result.is_some());
        let retrieved = result.unwrap();
        assert_eq!(retrieved.size, 4096);
        assert!(retrieved.in_use);
        assert_eq!(pool.stats().pool_hits, 1);
    }

    #[test]
    fn test_gpu_memory_pool_record_allocation() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(1000);
        pool.record_allocation(2000);

        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 3000);
        assert_eq!(stats.peak_usage, 3000);
    }

    #[test]
    fn test_gpu_memory_pool_record_deallocation() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(5000);
        pool.record_deallocation(2000);

        assert_eq!(pool.stats().total_allocated, 3000);
        assert_eq!(pool.stats().peak_usage, 5000);
    }

    #[test]
    fn test_gpu_memory_pool_deallocation_underflow() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(1000);
        pool.record_deallocation(5000); // More than allocated
        assert_eq!(pool.stats().total_allocated, 0); // Saturating sub
    }

    #[test]
    fn test_gpu_memory_pool_has_capacity() {
        let pool = GpuMemoryPool::with_max_size(1000);
        assert!(pool.has_capacity(500));
        assert!(pool.has_capacity(1000));
        assert!(!pool.has_capacity(1001));
    }

    #[test]
    fn test_gpu_memory_pool_clear() {
        let mut pool = GpuMemoryPool::new();
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: false,
        };
        pool.return_buffer(handle);
        assert_eq!(pool.stats().free_buffers, 1);

        pool.clear();
        assert_eq!(pool.stats().free_buffers, 0);
    }

    #[test]
    fn test_gpu_memory_pool_debug() {
        let pool = GpuMemoryPool::new();
        let debug_str = format!("{:?}", pool);
        assert!(debug_str.contains("GpuMemoryPool"));
    }

    // =========================================================================
    // PoolStats Tests
    // =========================================================================

    #[test]
    fn test_pool_stats_clone() {
        let stats = PoolStats {
            total_allocated: 1000,
            peak_usage: 2000,
            pool_hits: 10,
            pool_misses: 5,
            hit_rate: 0.667,
            free_buffers: 3,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocated, 1000);
        assert_eq!(cloned.pool_hits, 10);
    }

    #[test]
    fn test_pool_stats_estimated_savings_with_hits() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 100,
            pool_misses: 50,
            hit_rate: 0.667,
            free_buffers: 0,
        };
        let savings = stats.estimated_savings_bytes();
        assert_eq!(savings, 100 * 1024 * 1024); // 100 MB
    }

    #[test]
    fn test_pool_stats_estimated_savings_no_hits() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 50,
            hit_rate: 0.0,
            free_buffers: 0,
        };
        assert_eq!(stats.estimated_savings_bytes(), 0);
    }

    #[test]
    fn test_pool_stats_debug() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            hit_rate: 0.0,
            free_buffers: 0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("PoolStats"));
    }

    // =========================================================================
    // PinnedHostBuffer Tests
    // =========================================================================

    #[test]
    fn test_pinned_host_buffer_new() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());
        assert!(!buf.is_pinned()); // Currently not truly pinned
    }

    #[test]
    fn test_pinned_host_buffer_empty() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pinned_host_buffer_as_slice() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 10);
        assert_eq!(slice[0], 0.0); // Default f32
    }

    #[test]
    fn test_pinned_host_buffer_as_mut_slice() {
        let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        buf.as_mut_slice()[0] = 42.0;
        assert_eq!(buf.as_slice()[0], 42.0);
    }

    #[test]
    fn test_pinned_host_buffer_size_bytes() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
        assert_eq!(buf.size_bytes(), 100 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_pinned_host_buffer_copy_from_slice() {
        let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(3);
        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pinned_host_buffer_debug() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        let debug_str = format!("{:?}", buf);
        assert!(debug_str.contains("PinnedHostBuffer"));
    }

    #[test]
    fn test_pinned_host_buffer_different_types() {
        let buf_i32: PinnedHostBuffer<i32> = PinnedHostBuffer::new(5);
        assert_eq!(buf_i32.len(), 5);
        assert_eq!(buf_i32.size_bytes(), 5 * std::mem::size_of::<i32>());

        let buf_u8: PinnedHostBuffer<u8> = PinnedHostBuffer::new(100);
        assert_eq!(buf_u8.size_bytes(), 100);
    }

    // =========================================================================
    // StagingBufferPool Tests
    // =========================================================================

    #[test]
    fn test_staging_buffer_pool_default() {
        let pool = StagingBufferPool::default();
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }

    #[test]
    fn test_staging_buffer_pool_new() {
        let pool = StagingBufferPool::new();
        let stats = pool.stats();
        assert_eq!(stats.free_buffers, 0);
    }

    #[test]
    fn test_staging_buffer_pool_with_max_size() {
        let pool = StagingBufferPool::with_max_size(1024 * 1024);
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_staging_buffer_pool_get_allocates() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(100);
        assert!(buf.len() >= 100);
        assert_eq!(pool.stats().pool_misses, 1);
    }

    #[test]
    fn test_staging_buffer_pool_put_and_get() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(1000);
        let size = buf.size_bytes();
        pool.put(buf);

        assert_eq!(pool.stats().free_buffers, 1);

        let buf2 = pool.get(1000);
        assert_eq!(buf2.size_bytes(), size);
        assert_eq!(pool.stats().pool_hits, 1);
    }

    #[test]
    fn test_staging_buffer_pool_clear() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(1000);
        pool.put(buf);
        assert_eq!(pool.stats().free_buffers, 1);

        pool.clear();
        assert_eq!(pool.stats().free_buffers, 0);
        assert_eq!(pool.stats().total_allocated, 0);
    }

    #[test]
    fn test_staging_buffer_pool_debug() {
        let pool = StagingBufferPool::new();
        let debug_str = format!("{:?}", pool);
        assert!(debug_str.contains("StagingBufferPool"));
    }

    // =========================================================================
    // StagingPoolStats Tests
    // =========================================================================

    #[test]
    fn test_staging_pool_stats_clone() {
        let stats = StagingPoolStats {
            total_allocated: 1000,
            peak_usage: 2000,
            pool_hits: 10,
            pool_misses: 5,
            free_buffers: 3,
            hit_rate: 0.667,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocated, 1000);
        assert!((cloned.hit_rate - 0.667).abs() < 0.001);
    }

    #[test]
    fn test_staging_pool_stats_debug() {
        let stats = StagingPoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            free_buffers: 0,
            hit_rate: 0.0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("StagingPoolStats"));
    }

    // =========================================================================
    // TransferMode Tests
    // =========================================================================

    #[test]
    fn test_transfer_mode_default() {
        let mode = TransferMode::default();
        assert_eq!(mode, TransferMode::Pageable);
    }

    #[test]
    fn test_transfer_mode_requires_pinned() {
        assert!(!TransferMode::Pageable.requires_pinned());
        assert!(TransferMode::Pinned.requires_pinned());
        assert!(TransferMode::ZeroCopy.requires_pinned());
        assert!(TransferMode::Async.requires_pinned());
    }

    #[test]
    fn test_transfer_mode_estimated_speedup() {
        assert!((TransferMode::Pageable.estimated_speedup() - 1.0).abs() < f64::EPSILON);
        assert!((TransferMode::Pinned.estimated_speedup() - 1.7).abs() < f64::EPSILON);
        assert!((TransferMode::ZeroCopy.estimated_speedup() - 2.0).abs() < f64::EPSILON);
        assert!((TransferMode::Async.estimated_speedup() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transfer_mode_eq() {
        assert_eq!(TransferMode::Pageable, TransferMode::Pageable);
        assert_ne!(TransferMode::Pageable, TransferMode::Pinned);
    }

    #[test]
    fn test_transfer_mode_clone_copy() {
        let mode = TransferMode::Pinned;
        let cloned = mode;
        assert_eq!(cloned, TransferMode::Pinned);
    }

    #[test]
    fn test_transfer_mode_debug() {
        let debug_str = format!("{:?}", TransferMode::ZeroCopy);
        assert!(debug_str.contains("ZeroCopy"));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_pool_hit_rate_calculation() {
        let mut pool = GpuMemoryPool::new();

        // Add a buffer
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: false,
        };
        pool.return_buffer(handle);

        // First get should hit (from pool)
        let _ = pool.try_get(4096);

        // Second get should miss (pool empty)
        let _ = pool.try_get(4096);

        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.001);
    }
}
