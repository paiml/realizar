//! PagedAttention KV Cache Management
//!
//! Per spec §8.1: Efficient KV cache management based on vLLM's PagedAttention.
//! Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
//!
//! ## Key Features
//!
//! - **Physical Pages**: Fixed-size memory blocks for KV cache storage
//! - **Page Tables**: Logical to physical page mapping per sequence
//! - **Copy-on-Write**: Efficient prefix sharing between sequences
//! - **Dynamic Allocation**: Pages allocated on-demand during generation
//!
//! ## Memory Layout
//!
//! ```text
//! Physical Page (block_size tokens):
//! ┌─────────────────────────────────────────┐
//! │  K: [block_size, num_heads, head_dim]   │
//! │  V: [block_size, num_heads, head_dim]   │
//! └─────────────────────────────────────────┘
//! ```

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Error type for PagedKvCache operations
#[derive(Debug, Error)]
pub enum PagedCacheError {
    /// Out of memory - no free pages available
    #[error("Out of memory: need {needed} pages, have {available}")]
    OutOfMemory {
        /// Number of pages needed
        needed: usize,
        /// Number of pages available
        available: usize,
    },

    /// Sequence not found in page table
    #[error("Sequence not found: {0}")]
    SequenceNotFound(u64),

    /// Invalid page access
    #[error("Invalid page access: page {page_id} at offset {offset}")]
    InvalidPageAccess {
        /// Page ID accessed
        page_id: u64,
        /// Offset within page
        offset: usize,
    },

    /// Page table corruption
    #[error("Page table corruption for sequence {seq_id}")]
    PageTableCorruption {
        /// Sequence ID
        seq_id: u64,
    },
}

/// Unique sequence identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeqId(u64);

impl SeqId {
    /// Create a new unique sequence ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl Default for SeqId {
    fn default() -> Self {
        Self::new()
    }
}

/// Physical page identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageId(u64);

impl PageId {
    /// Create a new page ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// KV cache data for a single page
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Page identifier
    pub id: PageId,
    /// Key cache: [block_size, num_heads, head_dim]
    pub keys: Vec<f32>,
    /// Value cache: [block_size, num_heads, head_dim]
    pub values: Vec<f32>,
    /// Number of tokens currently stored in this page
    pub num_tokens: usize,
    /// Reference count for copy-on-write
    pub ref_count: usize,
}

impl KvPage {
    /// Create a new empty KV page
    pub fn new(id: PageId, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let page_size = block_size * num_heads * head_dim;
        Self {
            id,
            keys: vec![0.0; page_size],
            values: vec![0.0; page_size],
            num_tokens: 0,
            ref_count: 1,
        }
    }

    /// Check if page is full
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_tokens >= block_size
    }

    /// Check if page is shared (copy-on-write)
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self, block_size: usize) -> usize {
        block_size.saturating_sub(self.num_tokens)
    }
}

/// PagedAttention KV cache manager
/// Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
pub struct PagedKvCache {
    /// Physical pages (fixed-size blocks)
    physical_pages: Vec<KvPage>,
    /// Logical to physical page mapping (per sequence)
    page_tables: HashMap<SeqId, Vec<PageId>>,
    /// Free page list
    free_pages: VecDeque<PageId>,
    /// Tokens per page (block size)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Total pages allocated
    total_pages: usize,
    /// Statistics
    stats: PagedCacheStats,
}

/// Statistics for PagedKvCache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PagedCacheStats {
    /// Total sequences allocated
    pub sequences_allocated: u64,
    /// Total sequences freed
    pub sequences_freed: u64,
    /// Total pages allocated
    pub pages_allocated: u64,
    /// Total pages freed
    pub pages_freed: u64,
    /// Current active sequences
    pub active_sequences: u64,
    /// Current used pages
    pub used_pages: u64,
    /// Copy-on-write operations
    pub cow_operations: u64,
}

impl PagedKvCache {
    /// Create a new PagedKvCache
    ///
    /// # Arguments
    /// * `total_pages` - Total number of physical pages to allocate
    /// * `block_size` - Tokens per page (typically 16 or 32)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    pub fn new(total_pages: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let mut physical_pages = Vec::with_capacity(total_pages);
        let mut free_pages = VecDeque::with_capacity(total_pages);

        // Pre-allocate all pages
        for i in 0..total_pages {
            let page_id = PageId::new(i as u64);
            physical_pages.push(KvPage::new(page_id, block_size, num_heads, head_dim));
            free_pages.push_back(page_id);
        }

        Self {
            physical_pages,
            page_tables: HashMap::new(),
            free_pages,
            block_size,
            num_heads,
            head_dim,
            total_pages,
            stats: PagedCacheStats::default(),
        }
    }

    /// Allocate pages for a new sequence
    pub fn allocate_sequence(&mut self, num_tokens: usize) -> Result<SeqId, PagedCacheError> {
        let num_pages = self.tokens_to_pages(num_tokens);

        if self.free_pages.len() < num_pages {
            return Err(PagedCacheError::OutOfMemory {
                needed: num_pages,
                available: self.free_pages.len(),
            });
        }

        let seq_id = SeqId::new();
        let mut pages = Vec::with_capacity(num_pages);

        for _ in 0..num_pages {
            if let Some(page_id) = self.free_pages.pop_front() {
                // Reset the page
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.num_tokens = 0;
                page.ref_count = 1;
                pages.push(page_id);
            }
        }

        self.page_tables.insert(seq_id, pages);
        self.stats.sequences_allocated += 1;
        self.stats.pages_allocated += num_pages as u64;
        self.stats.active_sequences += 1;
        self.stats.used_pages += num_pages as u64;

        Ok(seq_id)
    }

    /// Extend sequence by allocating more pages for generation
    pub fn extend(&mut self, seq_id: SeqId, num_tokens: usize) -> Result<(), PagedCacheError> {
        // First, gather info without holding mutable borrow
        let (current_pages, current_tokens) = {
            let pages = self
                .page_tables
                .get(&seq_id)
                .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

            let mut total_tokens = 0;
            for page_id in pages {
                let page = &self.physical_pages[page_id.value() as usize];
                total_tokens += page.num_tokens;
            }
            (pages.len(), total_tokens)
        };

        let current_capacity = current_pages * self.block_size;
        let total_needed = current_tokens + num_tokens;

        if total_needed <= current_capacity {
            // No new pages needed
            return Ok(());
        }

        let additional_pages = self.tokens_to_pages(total_needed) - current_pages;

        if self.free_pages.len() < additional_pages {
            return Err(PagedCacheError::OutOfMemory {
                needed: additional_pages,
                available: self.free_pages.len(),
            });
        }

        // Collect new page IDs
        let mut new_pages = Vec::with_capacity(additional_pages);
        for _ in 0..additional_pages {
            if let Some(page_id) = self.free_pages.pop_front() {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.num_tokens = 0;
                page.ref_count = 1;
                new_pages.push(page_id);
            }
        }

        // Now update page table
        if let Some(pages) = self.page_tables.get_mut(&seq_id) {
            pages.extend(new_pages);
        }

        self.stats.pages_allocated += additional_pages as u64;
        self.stats.used_pages += additional_pages as u64;

        Ok(())
    }

    /// Free sequence and return pages to pool
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(pages) = self.page_tables.remove(&seq_id) {
            for page_id in pages {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.ref_count = page.ref_count.saturating_sub(1);

                // Only return to free list if no references remain
                if page.ref_count == 0 {
                    self.free_pages.push_back(page_id);
                    self.stats.pages_freed += 1;
                    self.stats.used_pages = self.stats.used_pages.saturating_sub(1);
                }
            }
            self.stats.sequences_freed += 1;
            self.stats.active_sequences = self.stats.active_sequences.saturating_sub(1);
        }
    }

    /// Fork a sequence (copy-on-write for prefix sharing)
    pub fn fork_sequence(&mut self, parent_seq_id: SeqId) -> Result<SeqId, PagedCacheError> {
        let parent_pages = self
            .page_tables
            .get(&parent_seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(parent_seq_id.value()))?
            .clone();

        // Increment reference counts for shared pages
        for page_id in &parent_pages {
            self.physical_pages[page_id.value() as usize].ref_count += 1;
        }

        let child_seq_id = SeqId::new();
        self.page_tables.insert(child_seq_id, parent_pages);

        self.stats.sequences_allocated += 1;
        self.stats.active_sequences += 1;
        self.stats.cow_operations += 1;

        Ok(child_seq_id)
    }

    /// Get the number of tokens stored for a sequence
    pub fn get_sequence_tokens(&self, seq_id: SeqId) -> Result<usize, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let mut total_tokens = 0;
        for page_id in pages {
            let page = &self.physical_pages[page_id.value() as usize];
            total_tokens += page.num_tokens;
        }

        Ok(total_tokens)
    }

    /// Update token count for sequence (after writing KV data)
    pub fn update_tokens(
        &mut self,
        seq_id: SeqId,
        num_tokens: usize,
    ) -> Result<(), PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let mut remaining = num_tokens;
        for page_id in pages {
            let page = &mut self.physical_pages[page_id.value() as usize];
            let tokens_in_page = remaining.min(self.block_size);
            page.num_tokens = tokens_in_page;
            remaining = remaining.saturating_sub(self.block_size);
            if remaining == 0 {
                break;
            }
        }

        Ok(())
    }

    /// Get physical page for a logical position
    pub fn get_page(
        &self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&KvPage, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let page_index = token_position / self.block_size;
        let page_id = pages
            .get(page_index)
            .ok_or(PagedCacheError::InvalidPageAccess {
                page_id: page_index as u64,
                offset: token_position,
            })?;

        Ok(&self.physical_pages[page_id.value() as usize])
    }

    /// Get mutable physical page (handles copy-on-write)
    pub fn get_page_mut(
        &mut self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&mut KvPage, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        let page_index = token_position / self.block_size;
        let page_id = *pages
            .get(page_index)
            .ok_or(PagedCacheError::InvalidPageAccess {
                page_id: page_index as u64,
                offset: token_position,
            })?;

        // Handle copy-on-write if page is shared
        let page = &self.physical_pages[page_id.value() as usize];
        if page.is_shared() {
            // Allocate a new page and copy data
            let new_page_id = self
                .free_pages
                .pop_front()
                .ok_or(PagedCacheError::OutOfMemory {
                    needed: 1,
                    available: 0,
                })?;

            // Copy data to new page
            let old_page = &self.physical_pages[page_id.value() as usize];
            let keys = old_page.keys.clone();
            let values = old_page.values.clone();
            let num_tokens = old_page.num_tokens;

            // Update old page ref count
            self.physical_pages[page_id.value() as usize].ref_count -= 1;

            // Setup new page
            let new_page = &mut self.physical_pages[new_page_id.value() as usize];
            new_page.keys = keys;
            new_page.values = values;
            new_page.num_tokens = num_tokens;
            new_page.ref_count = 1;

            // Update page table
            let pages = self
                .page_tables
                .get_mut(&seq_id)
                .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;
            pages[page_index] = new_page_id;

            self.stats.cow_operations += 1;
            self.stats.pages_allocated += 1;
            self.stats.used_pages += 1;

            return Ok(&mut self.physical_pages[new_page_id.value() as usize]);
        }

        Ok(&mut self.physical_pages[page_id.value() as usize])
    }

    /// Get cache statistics
    pub fn stats(&self) -> &PagedCacheStats {
        &self.stats
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2; // f32 = 4 bytes, K+V = 2
        self.stats.used_pages as usize * page_size
    }

    /// Get total capacity in bytes
    pub fn total_capacity(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2;
        self.total_pages * page_size
    }

    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_pages == 0 {
            return 0.0;
        }
        (self.stats.used_pages as f32 / self.total_pages as f32) * 100.0
    }

    /// Number of free pages available
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Number of pages needed for tokens
    fn tokens_to_pages(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === SeqId Tests ===

    #[test]
    fn test_seq_id_new() {
        let id1 = SeqId::new();
        let id2 = SeqId::new();
        assert_ne!(id1.value(), id2.value());
    }

    #[test]
    fn test_seq_id_default() {
        let id1 = SeqId::default();
        let id2 = SeqId::default();
        assert_ne!(id1, id2);
    }

    // === PageId Tests ===

    #[test]
    fn test_page_id_new() {
        let id = PageId::new(42);
        assert_eq!(id.value(), 42);
    }

    // === KvPage Tests ===

    #[test]
    fn test_kv_page_new() {
        let page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert_eq!(page.num_tokens, 0);
        assert_eq!(page.ref_count, 1);
        assert_eq!(page.keys.len(), 16 * 8 * 64);
        assert_eq!(page.values.len(), 16 * 8 * 64);
    }

    #[test]
    fn test_kv_page_is_full() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert!(!page.is_full(16));
        page.num_tokens = 16;
        assert!(page.is_full(16));
    }

    #[test]
    fn test_kv_page_is_shared() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert!(!page.is_shared());
        page.ref_count = 2;
        assert!(page.is_shared());
    }

    #[test]
    fn test_kv_page_remaining_capacity() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert_eq!(page.remaining_capacity(16), 16);
        page.num_tokens = 10;
        assert_eq!(page.remaining_capacity(16), 6);
    }

    // === PagedKvCache Tests ===

    #[test]
    fn test_paged_kv_cache_new() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_allocate_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).unwrap();

        // 32 tokens needs 2 pages (16 tokens per page)
        assert_eq!(cache.free_page_count(), 98);
        assert_eq!(cache.stats().active_sequences, 1);
        assert_eq!(cache.stats().pages_allocated, 2);
        // seq_id is valid (non-zero ID counter)
        assert!(seq_id.value() < u64::MAX);
    }

    #[test]
    fn test_allocate_sequence_out_of_memory() {
        let mut cache = PagedKvCache::new(1, 16, 8, 64);

        // First allocation succeeds
        let _ = cache.allocate_sequence(10).unwrap();

        // Second allocation fails
        let result = cache.allocate_sequence(20);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_extend_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).unwrap();

        // Initially 1 page
        assert_eq!(cache.free_page_count(), 99);

        // Extend to need 2 pages
        cache.extend(seq_id, 20).unwrap();
        assert_eq!(cache.free_page_count(), 98);
    }

    #[test]
    fn test_free_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).unwrap();

        assert_eq!(cache.free_page_count(), 98);

        cache.free_sequence(seq_id);

        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_fork_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).unwrap();

        let child_id = cache.fork_sequence(parent_id).unwrap();

        // Pages are shared via COW
        assert_eq!(cache.stats().active_sequences, 2);
        assert_eq!(cache.stats().cow_operations, 1);
        assert_ne!(parent_id, child_id);
    }

    #[test]
    fn test_get_page() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).unwrap();

        let page = cache.get_page(seq_id, 0).unwrap();
        assert_eq!(
            page.id.value(),
            cache.page_tables.get(&seq_id).unwrap()[0].value()
        );

        let page2 = cache.get_page(seq_id, 16).unwrap();
        assert_eq!(
            page2.id.value(),
            cache.page_tables.get(&seq_id).unwrap()[1].value()
        );
    }

    #[test]
    fn test_get_page_invalid() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).unwrap();

        let result = cache.get_page(seq_id, 100); // Beyond allocated pages
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    #[test]
    fn test_get_sequence_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).unwrap();
        cache.update_tokens(seq_id, 10).unwrap();

        let tokens = cache.get_sequence_tokens(seq_id).unwrap();
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_memory_usage() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        assert_eq!(cache.memory_usage(), 0);

        let _ = cache.allocate_sequence(16).unwrap();

        // 1 page * 16 tokens * 8 heads * 64 dim * 4 bytes * 2 (K+V)
        let expected = 16 * 8 * 64 * 4 * 2;
        assert_eq!(cache.memory_usage(), expected);
    }

    #[test]
    fn test_total_capacity() {
        let cache = PagedKvCache::new(100, 16, 8, 64);

        // 100 pages * 16 tokens * 8 heads * 64 dim * 4 bytes * 2 (K+V)
        let expected = 100 * 16 * 8 * 64 * 4 * 2;
        assert_eq!(cache.total_capacity(), expected);
    }

    #[test]
    fn test_utilization() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        assert_eq!(cache.utilization(), 0.0);

        let _ = cache.allocate_sequence(160).unwrap(); // 10 pages

        assert!((cache.utilization() - 10.0).abs() < 0.01);
    }

    // === Error Display Tests ===

    #[test]
    fn test_paged_cache_error_display() {
        let err = PagedCacheError::OutOfMemory {
            needed: 10,
            available: 5,
        };
        assert!(err.to_string().contains("need 10"));
        assert!(err.to_string().contains("have 5"));

        let err = PagedCacheError::SequenceNotFound(42);
        assert!(err.to_string().contains("42"));

        let err = PagedCacheError::InvalidPageAccess {
            page_id: 5,
            offset: 100,
        };
        assert!(err.to_string().contains("page 5"));

        let err = PagedCacheError::PageTableCorruption { seq_id: 99 };
        assert!(err.to_string().contains("99"));
    }

    // === Stats Tests ===

    #[test]
    fn test_paged_cache_stats_default() {
        let stats = PagedCacheStats::default();
        assert_eq!(stats.sequences_allocated, 0);
        assert_eq!(stats.sequences_freed, 0);
        assert_eq!(stats.pages_allocated, 0);
    }

    #[test]
    fn test_stats_serialization() {
        let stats = PagedCacheStats {
            sequences_allocated: 10,
            sequences_freed: 5,
            pages_allocated: 100,
            pages_freed: 50,
            active_sequences: 5,
            used_pages: 50,
            cow_operations: 3,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: PagedCacheStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.sequences_allocated, stats.sequences_allocated);
        assert_eq!(parsed.cow_operations, stats.cow_operations);
    }

    // === Copy-on-Write Tests ===

    #[test]
    fn test_cow_on_write() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).unwrap();
        cache.update_tokens(parent_id, 16).unwrap();

        // Fork creates shared pages
        let child_id = cache.fork_sequence(parent_id).unwrap();

        // Get mutable page should trigger COW
        let initial_cow = cache.stats().cow_operations;
        let _page = cache.get_page_mut(child_id, 0).unwrap();

        // COW should have been triggered
        assert!(cache.stats().cow_operations > initial_cow);
    }

    #[test]
    fn test_sequence_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_sequence_tokens(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }
}
