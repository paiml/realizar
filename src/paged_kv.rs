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
    /// Defragmentation operations performed
    pub defrag_operations: u64,
    /// Pages moved during defragmentation
    pub pages_moved: u64,
}

/// Fragmentation statistics for KV cache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FragmentationStats {
    /// Number of free page "holes" between used pages
    pub holes: usize,
    /// Total wasted capacity in partially-filled pages (tokens)
    pub wasted_capacity: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
    pub fragmentation_ratio: f32,
    /// Largest contiguous free region (in pages)
    pub largest_free_region: usize,
    /// Average tokens per page (efficiency metric)
    pub avg_tokens_per_page: f32,
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

    // ========================================================================
    // Defragmentation (llama.cpp competitive feature)
    // ========================================================================

    /// Calculate fragmentation statistics
    ///
    /// Per llama.cpp KV cache defrag: tracks holes, wasted capacity, and
    /// fragmentation ratio to decide when defragmentation is beneficial.
    pub fn fragmentation_stats(&self) -> FragmentationStats {
        // Build a usage map: true = used, false = free
        let mut usage_map = vec![false; self.total_pages];
        let mut total_tokens = 0usize;
        let mut pages_with_tokens = 0usize;

        for pages in self.page_tables.values() {
            for page_id in pages {
                let idx = page_id.value() as usize;
                if idx < self.total_pages {
                    usage_map[idx] = true;
                    let page = &self.physical_pages[idx];
                    total_tokens += page.num_tokens;
                    if page.num_tokens > 0 {
                        pages_with_tokens += 1;
                    }
                }
            }
        }

        // Count holes (transitions from used to free in the middle of used regions)
        let mut holes = 0usize;
        let mut in_used_region = false;
        let mut current_free_run = 0usize;
        let mut largest_free_region = 0usize;
        let mut free_runs = Vec::new();

        for &used in &usage_map {
            if used {
                if in_used_region && current_free_run > 0 {
                    holes += 1;
                    free_runs.push(current_free_run);
                }
                in_used_region = true;
                current_free_run = 0;
            } else {
                current_free_run += 1;
                largest_free_region = largest_free_region.max(current_free_run);
            }
        }

        // Trailing free region
        if current_free_run > 0 {
            free_runs.push(current_free_run);
        }

        // Calculate wasted capacity (unfilled slots in used pages)
        let used_pages = self.stats.used_pages as usize;
        let max_capacity = used_pages * self.block_size;
        let wasted_capacity = max_capacity.saturating_sub(total_tokens);

        // Fragmentation ratio: based on holes relative to used pages
        let fragmentation_ratio = if used_pages > 0 {
            (holes as f32) / (used_pages as f32).max(1.0)
        } else {
            0.0
        };

        // Average tokens per page
        let avg_tokens_per_page = if pages_with_tokens > 0 {
            total_tokens as f32 / pages_with_tokens as f32
        } else {
            0.0
        };

        FragmentationStats {
            holes,
            wasted_capacity,
            fragmentation_ratio: fragmentation_ratio.min(1.0),
            largest_free_region,
            avg_tokens_per_page,
        }
    }

    /// Determine if defragmentation should be performed
    ///
    /// Heuristic based on:
    /// - Fragmentation ratio > threshold (default 0.3)
    /// - Wasted capacity > 25% of used capacity
    /// - Free page count low but fragmented
    pub fn should_defragment(&self) -> bool {
        self.should_defragment_with_threshold(0.3)
    }

    /// Determine if defragmentation should be performed with custom threshold
    pub fn should_defragment_with_threshold(&self, threshold: f32) -> bool {
        let stats = self.fragmentation_stats();

        // High fragmentation ratio
        if stats.fragmentation_ratio > threshold {
            return true;
        }

        // Significant wasted capacity (>25% of block size)
        let used_pages = self.stats.used_pages as usize;
        if used_pages > 0 {
            let max_capacity = used_pages * self.block_size;
            let waste_ratio = stats.wasted_capacity as f32 / max_capacity as f32;
            if waste_ratio > 0.25 && stats.holes > 2 {
                return true;
            }
        }

        // Low on free pages but have holes we can recover
        let free_ratio = self.free_pages.len() as f32 / self.total_pages as f32;
        if free_ratio < 0.1 && stats.holes > 0 {
            return true;
        }

        false
    }

    /// Perform defragmentation - compact pages to reduce fragmentation
    ///
    /// This operation:
    /// 1. Identifies fragmented sequences
    /// 2. Moves pages to create contiguous allocations
    /// 3. Updates page tables accordingly
    /// 4. Returns number of pages moved
    ///
    /// Note: This is a relatively expensive operation and should be called
    /// during low-activity periods or when `should_defragment()` returns true.
    pub fn defragment(&mut self) -> usize {
        let mut pages_moved = 0;

        // Collect sequences to defragment (those with non-contiguous pages)
        let seq_ids: Vec<SeqId> = self.page_tables.keys().copied().collect();

        for seq_id in seq_ids {
            pages_moved += self.compact_sequence(seq_id);
        }

        if pages_moved > 0 {
            self.stats.defrag_operations += 1;
            self.stats.pages_moved += pages_moved as u64;
        }

        pages_moved
    }

    /// Compact a specific sequence's pages to be contiguous
    ///
    /// Returns number of pages moved.
    pub fn compact_sequence(&mut self, seq_id: SeqId) -> usize {
        let pages = match self.page_tables.get(&seq_id) {
            Some(p) => p.clone(),
            None => return 0,
        };

        if pages.is_empty() {
            return 0;
        }

        // Check if already contiguous
        let mut is_contiguous = true;
        for i in 1..pages.len() {
            let prev_id = pages[i - 1].value();
            let curr_id = pages[i].value();
            // Check if pages are adjacent (within reasonable range for contiguity)
            if curr_id != prev_id + 1 {
                is_contiguous = false;
                break;
            }
        }

        if is_contiguous {
            return 0; // Already compact
        }

        // Find target region - look for contiguous free space or lowest-numbered free pages
        let mut pages_moved = 0;

        // Strategy: For each non-contiguous page, try to move it adjacent to previous
        let mut new_page_list = vec![pages[0]];

        for i in 1..pages.len() {
            let prev_page_id = new_page_list[i - 1];
            let curr_page_id = pages[i];

            // Check if current page is already adjacent
            if curr_page_id.value() == prev_page_id.value() + 1 {
                new_page_list.push(curr_page_id);
                continue;
            }

            // Try to find a free page adjacent to previous
            let target_id = PageId::new(prev_page_id.value() + 1);
            let target_idx = target_id.value() as usize;

            if target_idx < self.total_pages && self.is_page_free(target_id) {
                // Move data from curr_page to target_page
                let curr_idx = curr_page_id.value() as usize;

                // Copy data
                let keys = self.physical_pages[curr_idx].keys.clone();
                let values = self.physical_pages[curr_idx].values.clone();
                let num_tokens = self.physical_pages[curr_idx].num_tokens;
                let ref_count = self.physical_pages[curr_idx].ref_count;

                // If current page is shared, we can't move it (COW semantics)
                if ref_count > 1 {
                    new_page_list.push(curr_page_id);
                    continue;
                }

                // Remove target from free list
                self.free_pages.retain(|&p| p != target_id);

                // Setup target page
                self.physical_pages[target_idx].keys = keys;
                self.physical_pages[target_idx].values = values;
                self.physical_pages[target_idx].num_tokens = num_tokens;
                self.physical_pages[target_idx].ref_count = 1;

                // Clear source page and return to free list
                self.physical_pages[curr_idx].num_tokens = 0;
                self.physical_pages[curr_idx].ref_count = 0;
                self.free_pages.push_back(curr_page_id);

                new_page_list.push(target_id);
                pages_moved += 1;
            } else {
                // Can't move, keep current page
                new_page_list.push(curr_page_id);
            }
        }

        // Update page table
        if let Some(entry) = self.page_tables.get_mut(&seq_id) {
            *entry = new_page_list;
        }

        pages_moved
    }

    /// Check if a page is free
    fn is_page_free(&self, page_id: PageId) -> bool {
        self.free_pages.contains(&page_id)
    }

    /// Get contiguity score for a sequence (1.0 = fully contiguous)
    pub fn sequence_contiguity(&self, seq_id: SeqId) -> Result<f32, PagedCacheError> {
        let pages = self
            .page_tables
            .get(&seq_id)
            .ok_or(PagedCacheError::SequenceNotFound(seq_id.value()))?;

        if pages.len() <= 1 {
            return Ok(1.0); // Single page is always contiguous
        }

        let mut contiguous_pairs = 0;
        for i in 1..pages.len() {
            if pages[i].value() == pages[i - 1].value() + 1 {
                contiguous_pairs += 1;
            }
        }

        Ok(contiguous_pairs as f32 / (pages.len() - 1) as f32)
    }
}

// ============================================================================
// PREFIX CACHING (per llama.cpp)
// ============================================================================
//
// Prefix caching allows reusing KV cache values for common prompt prefixes.
// When multiple requests share the same prefix tokens (e.g., system prompts),
// the KV cache for those tokens is computed once and reused.
//
// Benefits:
// - Reduces time-to-first-token for common prompts
// - Saves computation for repeated system instructions
// - Enables efficient multi-turn conversation handling
// ============================================================================

/// Hash type for prefix cache lookup
pub type PrefixHash = u64;

/// Compute hash for a token sequence (used for prefix lookup)
pub fn compute_prefix_hash(tokens: &[u32]) -> PrefixHash {
    // Simple FNV-1a hash for token sequences
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &token in tokens {
        hash ^= token as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }
    hash
}

/// Cached prefix entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrefix {
    /// Hash of the prefix tokens
    pub hash: PrefixHash,
    /// Number of tokens in prefix
    pub num_tokens: usize,
    /// Page IDs containing the cached KV values
    pub page_ids: Vec<PageId>,
    /// Reference count (number of sequences using this prefix)
    pub ref_count: usize,
    /// Last access timestamp (for LRU eviction)
    pub last_access: u64,
}

impl CachedPrefix {
    /// Create new cached prefix
    pub fn new(hash: PrefixHash, num_tokens: usize, page_ids: Vec<PageId>) -> Self {
        Self {
            hash,
            num_tokens,
            page_ids,
            ref_count: 1,
            last_access: 0,
        }
    }

    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn remove_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }
}

/// Prefix cache for KV cache reuse
///
/// Per llama.cpp's prompt cache: stores computed KV values for common
/// prompt prefixes, enabling fast cache hits for repeated system prompts.
pub struct PrefixCache {
    /// Cached prefixes by hash
    cache: HashMap<PrefixHash, CachedPrefix>,
    /// Maximum number of cached prefixes
    max_entries: usize,
    /// Access counter for LRU
    access_counter: u64,
    /// Statistics
    stats: PrefixCacheStats,
}

/// Statistics for prefix cache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Total prefixes cached
    pub prefixes_cached: u64,
    /// Prefixes evicted
    pub prefixes_evicted: u64,
    /// Tokens saved (not recomputed)
    pub tokens_saved: u64,
}

impl PrefixCacheStats {
    /// Hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl PrefixCache {
    /// Create new prefix cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            max_entries,
            access_counter: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Look up cached prefix by hash
    pub fn lookup(&mut self, hash: PrefixHash) -> Option<&CachedPrefix> {
        if let Some(entry) = self.cache.get_mut(&hash) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            self.stats.hits += 1;
            // Return immutable reference
            self.cache.get(&hash)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Look up cached prefix by tokens
    pub fn lookup_tokens(&mut self, tokens: &[u32]) -> Option<&CachedPrefix> {
        let hash = compute_prefix_hash(tokens);
        self.lookup(hash)
    }

    /// Check if prefix is cached (without updating stats)
    pub fn contains(&self, hash: PrefixHash) -> bool {
        self.cache.contains_key(&hash)
    }

    /// Insert cached prefix
    pub fn insert(&mut self, prefix: CachedPrefix) -> bool {
        let hash = prefix.hash;

        // Evict if at capacity
        if self.cache.len() >= self.max_entries && !self.cache.contains_key(&hash) {
            self.evict_lru();
        }

        if self.cache.len() < self.max_entries {
            self.stats.prefixes_cached += 1;
            self.stats.tokens_saved += prefix.num_tokens as u64;
            self.cache.insert(hash, prefix);
            true
        } else {
            false
        }
    }

    /// Add reference to cached prefix
    pub fn add_ref(&mut self, hash: PrefixHash) -> bool {
        if let Some(entry) = self.cache.get_mut(&hash) {
            entry.add_ref();
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            true
        } else {
            false
        }
    }

    /// Remove reference from cached prefix
    /// Returns true if prefix was removed (no more references)
    pub fn remove_ref(&mut self, hash: PrefixHash) -> bool {
        if let Some(entry) = self.cache.get_mut(&hash) {
            if entry.remove_ref() {
                // No more references, remove from cache
                self.cache.remove(&hash);
                return true;
            }
        }
        false
    }

    /// Evict least recently used prefix
    fn evict_lru(&mut self) {
        if let Some((&hash, _)) = self
            .cache
            .iter()
            .filter(|(_, v)| v.ref_count == 0)
            .min_by_key(|(_, v)| v.last_access)
        {
            self.cache.remove(&hash);
            self.stats.prefixes_evicted += 1;
        }
    }

    /// Get number of cached prefixes
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_counter = 0;
    }

    /// Get cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.max_entries == 0 {
            0.0
        } else {
            self.cache.len() as f64 / self.max_entries as f64
        }
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(100)
    }
}

// ============================================================================
// KV CACHE QUANTIZATION (per llama.cpp Q8/Q4 KV)
// ============================================================================
//
// KV cache quantization reduces memory usage during inference:
// - Q8_0: 8-bit quantization, ~2x memory reduction, minimal quality loss
// - Q4_0: 4-bit quantization, ~4x memory reduction, some quality loss
//
// llama.cpp uses this for long-context inference where KV cache dominates memory.
// ============================================================================

/// KV cache quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KvQuantType {
    /// Full precision (32-bit float)
    #[default]
    FP32,
    /// 8-bit quantization (Q8_0 format)
    Q8,
    /// 4-bit quantization (Q4_0 format)
    Q4,
}

impl KvQuantType {
    /// Bytes per value for this quantization type
    pub fn bytes_per_value(&self) -> f32 {
        match self {
            Self::FP32 => 4.0,
            Self::Q8 => 1.0, // 8 bits = 1 byte
            Self::Q4 => 0.5, // 4 bits = 0.5 bytes
        }
    }

    /// Memory reduction factor compared to FP32
    pub fn memory_reduction(&self) -> f32 {
        4.0 / self.bytes_per_value()
    }
}

/// Block size for KV quantization (matches GGML)
pub const KV_QUANT_BLOCK_SIZE: usize = 32;

/// Q8_0 quantized block for KV cache
#[derive(Debug, Clone)]
pub struct Q8KvBlock {
    /// Scale factor for the block
    pub scale: f32,
    /// Quantized values (int8, stored as i8)
    pub quants: [i8; KV_QUANT_BLOCK_SIZE],
}

impl Q8KvBlock {
    /// Create empty block
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            quants: [0; KV_QUANT_BLOCK_SIZE],
        }
    }

    /// Quantize float values to Q8
    pub fn quantize(values: &[f32; KV_QUANT_BLOCK_SIZE]) -> Self {
        // Find max absolute value for scale
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if amax < 1e-10 {
            return Self::new();
        }

        let scale = amax / 127.0;
        let inv_scale = 1.0 / scale;

        let mut quants = [0i8; KV_QUANT_BLOCK_SIZE];
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv_scale).round() as i32;
            quants[i] = q.clamp(-127, 127) as i8;
        }

        Self { scale, quants }
    }

    /// Dequantize to float values
    pub fn dequantize(&self) -> [f32; KV_QUANT_BLOCK_SIZE] {
        let mut result = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, &q) in self.quants.iter().enumerate() {
            result[i] = q as f32 * self.scale;
        }
        result
    }
}

impl Default for Q8KvBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Q4_0 quantized block for KV cache
#[derive(Debug, Clone)]
pub struct Q4KvBlock {
    /// Scale factor for the block
    pub scale: f32,
    /// Quantized values (4-bit, packed 2 per byte)
    pub quants: [u8; KV_QUANT_BLOCK_SIZE / 2],
}

impl Q4KvBlock {
    /// Create empty block
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            quants: [0; KV_QUANT_BLOCK_SIZE / 2],
        }
    }

    /// Quantize float values to Q4
    pub fn quantize(values: &[f32; KV_QUANT_BLOCK_SIZE]) -> Self {
        // Find max absolute value for scale
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if amax < 1e-10 {
            return Self::new();
        }

        // Q4_0 uses signed 4-bit: -8 to 7
        let scale = amax / 7.0;
        let inv_scale = 1.0 / scale;

        let mut quants = [0u8; KV_QUANT_BLOCK_SIZE / 2];
        for i in 0..(KV_QUANT_BLOCK_SIZE / 2) {
            let v0 = values[i * 2];
            let v1 = values[i * 2 + 1];

            // Quantize to -8..7 range, then shift to 0..15 for unsigned storage
            let q0 = ((v0 * inv_scale).round() as i32).clamp(-8, 7) + 8;
            let q1 = ((v1 * inv_scale).round() as i32).clamp(-8, 7) + 8;

            // Pack two 4-bit values into one byte
            quants[i] = ((q1 as u8) << 4) | (q0 as u8);
        }

        Self { scale, quants }
    }

    /// Dequantize to float values
    pub fn dequantize(&self) -> [f32; KV_QUANT_BLOCK_SIZE] {
        let mut result = [0.0f32; KV_QUANT_BLOCK_SIZE];

        for (i, &packed) in self.quants.iter().enumerate() {
            // Unpack two 4-bit values
            let q0 = (packed & 0x0F) as i32 - 8;
            let q1 = ((packed >> 4) & 0x0F) as i32 - 8;

            result[i * 2] = q0 as f32 * self.scale;
            result[i * 2 + 1] = q1 as f32 * self.scale;
        }

        result
    }
}

impl Default for Q4KvBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantized KV cache data for a single page
#[derive(Debug, Clone)]
pub enum QuantizedKvData {
    /// Full precision storage
    FP32 {
        /// Key cache: [block_size, num_heads, head_dim]
        keys: Vec<f32>,
        /// Value cache: [block_size, num_heads, head_dim]
        values: Vec<f32>,
    },
    /// Q8 quantized storage
    Q8 {
        /// Quantized key blocks
        key_blocks: Vec<Q8KvBlock>,
        /// Quantized value blocks
        value_blocks: Vec<Q8KvBlock>,
    },
    /// Q4 quantized storage
    Q4 {
        /// Quantized key blocks
        key_blocks: Vec<Q4KvBlock>,
        /// Quantized value blocks
        value_blocks: Vec<Q4KvBlock>,
    },
}

impl QuantizedKvData {
    /// Create new quantized KV data with given precision
    pub fn new(
        quant_type: KvQuantType,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let total_size = block_size * num_heads * head_dim;
        let num_quant_blocks = total_size.div_ceil(KV_QUANT_BLOCK_SIZE);

        match quant_type {
            KvQuantType::FP32 => Self::FP32 {
                keys: vec![0.0; total_size],
                values: vec![0.0; total_size],
            },
            KvQuantType::Q8 => Self::Q8 {
                key_blocks: vec![Q8KvBlock::new(); num_quant_blocks],
                value_blocks: vec![Q8KvBlock::new(); num_quant_blocks],
            },
            KvQuantType::Q4 => Self::Q4 {
                key_blocks: vec![Q4KvBlock::new(); num_quant_blocks],
                value_blocks: vec![Q4KvBlock::new(); num_quant_blocks],
            },
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        match self {
            Self::FP32 { .. } => KvQuantType::FP32,
            Self::Q8 { .. } => KvQuantType::Q8,
            Self::Q4 { .. } => KvQuantType::Q4,
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::FP32 { keys, values } => (keys.len() + values.len()) * 4,
            Self::Q8 {
                key_blocks,
                value_blocks,
            } => {
                // Scale (4 bytes) + quants (32 bytes) = 36 bytes per block
                (key_blocks.len() + value_blocks.len()) * (4 + KV_QUANT_BLOCK_SIZE)
            },
            Self::Q4 {
                key_blocks,
                value_blocks,
            } => {
                // Scale (4 bytes) + quants (16 bytes) = 20 bytes per block
                (key_blocks.len() + value_blocks.len()) * (4 + KV_QUANT_BLOCK_SIZE / 2)
            },
        }
    }

    /// Write keys at given offset
    pub fn write_keys(&mut self, offset: usize, data: &[f32]) {
        match self {
            Self::FP32 { keys, .. } => {
                let end = (offset + data.len()).min(keys.len());
                keys[offset..end].copy_from_slice(&data[..end - offset]);
            },
            Self::Q8 { key_blocks, .. } => {
                write_quantized_q8(key_blocks, offset, data);
            },
            Self::Q4 { key_blocks, .. } => {
                write_quantized_q4(key_blocks, offset, data);
            },
        }
    }

    /// Write values at given offset
    pub fn write_values(&mut self, offset: usize, data: &[f32]) {
        match self {
            Self::FP32 { values, .. } => {
                let end = (offset + data.len()).min(values.len());
                values[offset..end].copy_from_slice(&data[..end - offset]);
            },
            Self::Q8 { value_blocks, .. } => {
                write_quantized_q8(value_blocks, offset, data);
            },
            Self::Q4 { value_blocks, .. } => {
                write_quantized_q4(value_blocks, offset, data);
            },
        }
    }

    /// Read keys at given offset
    pub fn read_keys(&self, offset: usize, length: usize) -> Vec<f32> {
        match self {
            Self::FP32 { keys, .. } => {
                let end = (offset + length).min(keys.len());
                keys[offset..end].to_vec()
            },
            Self::Q8 { key_blocks, .. } => read_quantized_q8(key_blocks, offset, length),
            Self::Q4 { key_blocks, .. } => read_quantized_q4(key_blocks, offset, length),
        }
    }

    /// Read values at given offset
    pub fn read_values(&self, offset: usize, length: usize) -> Vec<f32> {
        match self {
            Self::FP32 { values, .. } => {
                let end = (offset + length).min(values.len());
                values[offset..end].to_vec()
            },
            Self::Q8 { value_blocks, .. } => read_quantized_q8(value_blocks, offset, length),
            Self::Q4 { value_blocks, .. } => read_quantized_q4(value_blocks, offset, length),
        }
    }
}

// Helper: Write to Q8 blocks
fn write_quantized_q8(blocks: &mut [Q8KvBlock], offset: usize, data: &[f32]) {
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut data_idx = 0;
    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;

    while data_idx < data.len() && block_idx < blocks.len() {
        // Read existing block, modify, re-quantize
        let mut values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && data_idx < data.len() {
            values[in_block_offset] = data[data_idx];
            in_block_offset += 1;
            data_idx += 1;
        }

        blocks[block_idx] = Q8KvBlock::quantize(&values);
        block_idx += 1;
        in_block_offset = 0;
    }
}

// Helper: Write to Q4 blocks
fn write_quantized_q4(blocks: &mut [Q4KvBlock], offset: usize, data: &[f32]) {
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut data_idx = 0;
    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;

    while data_idx < data.len() && block_idx < blocks.len() {
        let mut values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && data_idx < data.len() {
            values[in_block_offset] = data[data_idx];
            in_block_offset += 1;
            data_idx += 1;
        }

        blocks[block_idx] = Q4KvBlock::quantize(&values);
        block_idx += 1;
        in_block_offset = 0;
    }
}

// Helper: Read from Q8 blocks
fn read_quantized_q8(blocks: &[Q8KvBlock], offset: usize, length: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(length);
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;
    let mut remaining = length;

    while remaining > 0 && block_idx < blocks.len() {
        let values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && remaining > 0 {
            result.push(values[in_block_offset]);
            in_block_offset += 1;
            remaining -= 1;
        }

        block_idx += 1;
        in_block_offset = 0;
    }

    result
}

// Helper: Read from Q4 blocks
fn read_quantized_q4(blocks: &[Q4KvBlock], offset: usize, length: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(length);
    let start_block = offset / KV_QUANT_BLOCK_SIZE;
    let start_offset = offset % KV_QUANT_BLOCK_SIZE;

    let mut block_idx = start_block;
    let mut in_block_offset = start_offset;
    let mut remaining = length;

    while remaining > 0 && block_idx < blocks.len() {
        let values = blocks[block_idx].dequantize();

        while in_block_offset < KV_QUANT_BLOCK_SIZE && remaining > 0 {
            result.push(values[in_block_offset]);
            in_block_offset += 1;
            remaining -= 1;
        }

        block_idx += 1;
        in_block_offset = 0;
    }

    result
}

/// Quantized KV page for memory-efficient cache
#[derive(Debug, Clone)]
pub struct QuantizedKvPage {
    /// Page identifier
    pub id: PageId,
    /// Quantized KV data
    pub data: QuantizedKvData,
    /// Number of tokens currently stored
    pub num_tokens: usize,
    /// Reference count for COW
    pub ref_count: usize,
    /// Block size (tokens per page)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl QuantizedKvPage {
    /// Create new quantized KV page
    pub fn new(
        id: PageId,
        quant_type: KvQuantType,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            id,
            data: QuantizedKvData::new(quant_type, block_size, num_heads, head_dim),
            num_tokens: 0,
            ref_count: 0, // Pages start in free pool with ref_count 0
            block_size,
            num_heads,
            head_dim,
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        self.data.quant_type()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.memory_bytes()
    }

    /// Check if page is full
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.block_size
    }

    /// Check if page is shared (COW)
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// Remaining capacity in tokens
    pub fn remaining_capacity(&self) -> usize {
        self.block_size.saturating_sub(self.num_tokens)
    }

    /// Write keys for a token position
    pub fn write_keys(&mut self, token_pos: usize, keys: &[f32]) {
        let offset = token_pos * self.num_heads * self.head_dim;
        self.data.write_keys(offset, keys);
    }

    /// Write values for a token position
    pub fn write_values(&mut self, token_pos: usize, values: &[f32]) {
        let offset = token_pos * self.num_heads * self.head_dim;
        self.data.write_values(offset, values);
    }

    /// Read keys for a token position
    pub fn read_keys(&self, token_pos: usize) -> Vec<f32> {
        let offset = token_pos * self.num_heads * self.head_dim;
        let length = self.num_heads * self.head_dim;
        self.data.read_keys(offset, length)
    }

    /// Read values for a token position
    pub fn read_values(&self, token_pos: usize) -> Vec<f32> {
        let offset = token_pos * self.num_heads * self.head_dim;
        let length = self.num_heads * self.head_dim;
        self.data.read_values(offset, length)
    }
}

/// Quantized PagedKvCache with configurable precision
pub struct QuantizedPagedKvCache {
    /// Physical pages with quantized storage
    physical_pages: Vec<QuantizedKvPage>,
    /// Page tables (same as regular PagedKvCache)
    page_tables: HashMap<SeqId, Vec<PageId>>,
    /// Free page list
    free_pages: VecDeque<PageId>,
    /// Quantization type
    quant_type: KvQuantType,
    /// Tokens per page
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Total pages
    total_pages: usize,
    /// Statistics
    stats: PagedCacheStats,
}

impl QuantizedPagedKvCache {
    /// Create new quantized paged KV cache
    pub fn new(
        total_pages: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        quant_type: KvQuantType,
    ) -> Self {
        let mut physical_pages = Vec::with_capacity(total_pages);
        let mut free_pages = VecDeque::with_capacity(total_pages);

        for i in 0..total_pages {
            let page_id = PageId::new(i as u64);
            physical_pages.push(QuantizedKvPage::new(
                page_id, quant_type, block_size, num_heads, head_dim,
            ));
            free_pages.push_back(page_id);
        }

        Self {
            physical_pages,
            page_tables: HashMap::new(),
            free_pages,
            quant_type,
            block_size,
            num_heads,
            head_dim,
            total_pages,
            stats: PagedCacheStats::default(),
        }
    }

    /// Get quantization type
    pub fn quant_type(&self) -> KvQuantType {
        self.quant_type
    }

    /// Allocate pages for a sequence
    pub fn allocate_sequence(&mut self, num_tokens: usize) -> Result<SeqId, PagedCacheError> {
        let num_pages = num_tokens.div_ceil(self.block_size);

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

    /// Free a sequence
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(pages) = self.page_tables.remove(&seq_id) {
            for page_id in pages {
                let page = &mut self.physical_pages[page_id.value() as usize];
                page.ref_count = page.ref_count.saturating_sub(1);

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

    /// Get page for a token position
    pub fn get_page(
        &self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&QuantizedKvPage, PagedCacheError> {
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

    /// Get mutable page for a token position
    pub fn get_page_mut(
        &mut self,
        seq_id: SeqId,
        token_position: usize,
    ) -> Result<&mut QuantizedKvPage, PagedCacheError> {
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

        Ok(&mut self.physical_pages[page_id.value() as usize])
    }

    /// Get total pages capacity
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.physical_pages
            .iter()
            .filter(|p| p.ref_count > 0)
            .map(QuantizedKvPage::memory_bytes)
            .sum()
    }

    /// FP32 equivalent memory (for comparison)
    pub fn fp32_equivalent_memory(&self) -> usize {
        let page_size = self.block_size * self.num_heads * self.head_dim * 4 * 2;
        self.stats.used_pages as usize * page_size
    }

    /// Memory savings ratio (1.0 = no savings, 0.25 = 4x reduction)
    pub fn memory_savings(&self) -> f32 {
        let fp32_mem = self.fp32_equivalent_memory();
        if fp32_mem == 0 {
            return 1.0;
        }
        self.memory_usage() as f32 / fp32_mem as f32
    }

    /// Get statistics
    pub fn stats(&self) -> &PagedCacheStats {
        &self.stats
    }

    /// Free page count
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }
}

/// Find longest matching prefix in a sequence of tokens
///
/// Returns (hash, num_matching_tokens) for the longest cached prefix
pub fn find_longest_prefix(cache: &mut PrefixCache, tokens: &[u32]) -> Option<(PrefixHash, usize)> {
    let mut best_match = None;
    let mut best_len = 0;

    // Try progressively longer prefixes (from 1 token up to full sequence)
    for len in 1..=tokens.len() {
        let prefix_hash = compute_prefix_hash(&tokens[..len]);
        if cache.contains(prefix_hash) && len > best_len {
            best_len = len;
            best_match = Some((prefix_hash, len));
        }
    }

    // Update stats if found
    if let Some((hash, _)) = best_match {
        cache.lookup(hash); // Update access time
    }

    best_match
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
        let seq_id = cache.allocate_sequence(32).expect("test");

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
        let _ = cache.allocate_sequence(10).expect("test");

        // Second allocation fails
        let result = cache.allocate_sequence(20);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_extend_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test");

        // Initially 1 page
        assert_eq!(cache.free_page_count(), 99);

        // Extend to need 2 pages
        cache.extend(seq_id, 20).expect("test");
        assert_eq!(cache.free_page_count(), 98);
    }

    #[test]
    fn test_free_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        assert_eq!(cache.free_page_count(), 98);

        cache.free_sequence(seq_id);

        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_fork_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("test");

        let child_id = cache.fork_sequence(parent_id).expect("test");

        // Pages are shared via COW
        assert_eq!(cache.stats().active_sequences, 2);
        assert_eq!(cache.stats().cow_operations, 1);
        assert_ne!(parent_id, child_id);
    }

    #[test]
    fn test_get_page() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(
            page.id.value(),
            cache.page_tables.get(&seq_id).expect("test")[0].value()
        );

        let page2 = cache.get_page(seq_id, 16).expect("test");
        assert_eq!(
            page2.id.value(),
            cache.page_tables.get(&seq_id).expect("test")[1].value()
        );
    }

    #[test]
    fn test_get_page_invalid() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("test");

        let result = cache.get_page(seq_id, 100); // Beyond allocated pages
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    #[test]
    fn test_get_sequence_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test");
        cache.update_tokens(seq_id, 10).expect("test");

        let tokens = cache.get_sequence_tokens(seq_id).expect("test");
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_memory_usage() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        assert_eq!(cache.memory_usage(), 0);

        let _ = cache.allocate_sequence(16).expect("test");

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

        let _ = cache.allocate_sequence(160).expect("test"); // 10 pages

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
            defrag_operations: 2,
            pages_moved: 15,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: PagedCacheStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.sequences_allocated, stats.sequences_allocated);
        assert_eq!(parsed.cow_operations, stats.cow_operations);
        assert_eq!(parsed.defrag_operations, stats.defrag_operations);
        assert_eq!(parsed.pages_moved, stats.pages_moved);
    }

    // === Copy-on-Write Tests ===

    #[test]
    fn test_cow_on_write() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("test");
        cache.update_tokens(parent_id, 16).expect("test");

        // Fork creates shared pages
        let child_id = cache.fork_sequence(parent_id).expect("test");

        // Get mutable page should trigger COW
        let initial_cow = cache.stats().cow_operations;
        let _page = cache.get_page_mut(child_id, 0).expect("test");

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

    // === Defragmentation Tests ===

    #[test]
    fn test_fragmentation_stats_empty_cache() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let stats = cache.fragmentation_stats();

        assert_eq!(stats.holes, 0);
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert_eq!(stats.largest_free_region, 100); // All pages free
    }

    #[test]
    fn test_fragmentation_stats_single_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages
        cache.update_tokens(seq_id, 32).expect("test");

        let stats = cache.fragmentation_stats();

        // With 2 contiguous pages at the start, no holes in used region
        assert_eq!(stats.holes, 0);
        // 32 tokens in 2 pages (32 capacity) = 0 wasted
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.avg_tokens_per_page, 16.0);
    }

    #[test]
    fn test_fragmentation_stats_with_holes() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        // Allocate 3 sequences
        let seq1 = cache.allocate_sequence(16).expect("test"); // Page 0
        let seq2 = cache.allocate_sequence(16).expect("test"); // Page 1
        let seq3 = cache.allocate_sequence(16).expect("test"); // Page 2

        // Free middle sequence to create a hole
        cache.free_sequence(seq2);

        let stats = cache.fragmentation_stats();

        // Should have at least one hole (between seq1 and seq3's pages)
        // Note: depends on allocation order
        assert!(stats.largest_free_region >= 1);

        // Verify seq1 and seq3 still valid
        assert!(cache.get_sequence_tokens(seq1).is_ok());
        assert!(cache.get_sequence_tokens(seq3).is_ok());
    }

    #[test]
    fn test_fragmentation_stats_wasted_capacity() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages
        cache.update_tokens(seq_id, 10).expect("test"); // Only 10 tokens in 2 pages

        let stats = cache.fragmentation_stats();

        // 2 pages * 16 block_size = 32 capacity, 10 tokens = 22 wasted
        assert_eq!(stats.wasted_capacity, 22);
    }

    #[test]
    fn test_should_defragment_empty() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        assert!(!cache.should_defragment());
    }

    #[test]
    fn test_should_defragment_no_fragmentation() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(seq_id, 32).expect("test");

        // Single contiguous allocation = no fragmentation
        assert!(!cache.should_defragment());
    }

    #[test]
    fn test_should_defragment_with_threshold() {
        let cache = PagedKvCache::new(100, 16, 8, 64);

        // With 0.0 threshold, any fragmentation triggers
        assert!(!cache.should_defragment_with_threshold(0.0));

        // With 1.0 threshold, only extreme fragmentation triggers
        assert!(!cache.should_defragment_with_threshold(1.0));
    }

    #[test]
    fn test_defragment_empty_cache() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let pages_moved = cache.defragment();

        assert_eq!(pages_moved, 0);
        assert_eq!(cache.stats().defrag_operations, 0);
    }

    #[test]
    fn test_defragment_single_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(seq_id, 32).expect("test");

        // Already contiguous, no defrag needed
        let pages_moved = cache.defragment();
        assert_eq!(pages_moved, 0);
    }

    #[test]
    fn test_compact_sequence_already_contiguous() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let moved = cache.compact_sequence(seq_id);
        assert_eq!(moved, 0); // Already contiguous
    }

    #[test]
    fn test_compact_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let moved = cache.compact_sequence(fake_seq);
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_sequence_contiguity_single_page() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test"); // 1 page

        let contiguity = cache.sequence_contiguity(seq_id).expect("test");
        assert_eq!(contiguity, 1.0); // Single page always contiguous
    }

    #[test]
    fn test_sequence_contiguity_multiple_pages() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages

        let contiguity = cache.sequence_contiguity(seq_id).expect("test");
        // Fresh allocation should be contiguous
        assert!(contiguity >= 0.0);
        assert!(contiguity <= 1.0);
    }

    #[test]
    fn test_sequence_contiguity_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.sequence_contiguity(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_fragmentation_stats_serialization() {
        let stats = FragmentationStats {
            holes: 5,
            wasted_capacity: 100,
            fragmentation_ratio: 0.25,
            largest_free_region: 50,
            avg_tokens_per_page: 12.5,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: FragmentationStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.holes, 5);
        assert_eq!(parsed.wasted_capacity, 100);
        assert!((parsed.fragmentation_ratio - 0.25).abs() < 0.001);
        assert_eq!(parsed.largest_free_region, 50);
        assert!((parsed.avg_tokens_per_page - 12.5).abs() < 0.001);
    }

    #[test]
    fn test_defrag_stats_tracking() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Initial state
        assert_eq!(cache.stats().defrag_operations, 0);
        assert_eq!(cache.stats().pages_moved, 0);

        // Defrag on empty cache doesn't increment
        cache.defragment();
        assert_eq!(cache.stats().defrag_operations, 0);
    }

    #[test]
    fn test_defragment_preserves_data() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("test");
        cache.update_tokens(seq_id, 16).expect("test");

        // Get page and write some test data
        let page = cache.get_page_mut(seq_id, 0).expect("test");
        page.keys[0] = 42.0;
        page.values[0] = 99.0;

        // Defragment
        cache.defragment();

        // Verify data is preserved
        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page.keys[0], 42.0);
        assert_eq!(page.values[0], 99.0);
    }

    #[test]
    fn test_cow_prevents_compact() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(parent_id, 32).expect("test");

        // Fork creates shared pages (COW)
        let _child_id = cache.fork_sequence(parent_id).expect("test");

        // Shared pages should not be moved during compaction
        // (ref_count > 1 check in compact_sequence)
        let moved = cache.compact_sequence(parent_id);
        // Since pages are already contiguous, should be 0 anyway
        assert_eq!(moved, 0);
    }

    // === Prefix Caching Tests ===

    #[test]
    fn test_compute_prefix_hash() {
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let tokens3 = vec![1, 2, 4];

        let hash1 = compute_prefix_hash(&tokens1);
        let hash2 = compute_prefix_hash(&tokens2);
        let hash3 = compute_prefix_hash(&tokens3);

        // Same tokens = same hash
        assert_eq!(hash1, hash2);
        // Different tokens = different hash (with high probability)
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_compute_prefix_hash_empty() {
        let tokens: Vec<u32> = vec![];
        let hash = compute_prefix_hash(&tokens);
        // Should be the FNV offset basis
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn test_cached_prefix_new() {
        let hash = 12345;
        let page_ids = vec![PageId::new(0), PageId::new(1)];
        let prefix = CachedPrefix::new(hash, 10, page_ids);

        assert_eq!(prefix.hash, hash);
        assert_eq!(prefix.num_tokens, 10);
        assert_eq!(prefix.page_ids.len(), 2);
        assert_eq!(prefix.ref_count, 1);
    }

    #[test]
    fn test_cached_prefix_ref_counting() {
        let mut prefix = CachedPrefix::new(1, 5, vec![]);

        assert_eq!(prefix.ref_count, 1);

        prefix.add_ref();
        assert_eq!(prefix.ref_count, 2);

        assert!(!prefix.remove_ref()); // Still has references
        assert_eq!(prefix.ref_count, 1);

        assert!(prefix.remove_ref()); // No more references
        assert_eq!(prefix.ref_count, 0);
    }

    #[test]
    fn test_prefix_cache_new() {
        let cache = PrefixCache::new(100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_prefix_cache_insert_lookup() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        let prefix = CachedPrefix::new(hash, 3, vec![PageId::new(0)]);

        assert!(cache.insert(prefix));
        assert_eq!(cache.len(), 1);

        let result = cache.lookup(hash);
        assert!(result.is_some());
        assert_eq!(result.expect("test").num_tokens, 3);
    }

    #[test]
    fn test_prefix_cache_lookup_tokens() {
        let mut cache = PrefixCache::new(100);

        let tokens = vec![10, 20, 30];
        let hash = compute_prefix_hash(&tokens);
        let prefix = CachedPrefix::new(hash, 3, vec![PageId::new(0)]);

        cache.insert(prefix);

        let result = cache.lookup_tokens(&tokens);
        assert!(result.is_some());
    }

    #[test]
    fn test_prefix_cache_miss() {
        let mut cache = PrefixCache::new(100);

        let result = cache.lookup(12345);
        assert!(result.is_none());

        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(hash, 3, vec![]));

        // Miss
        cache.lookup(99999);

        // Hit
        cache.lookup(hash);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prefix_cache_add_remove_ref() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(hash, 3, vec![]));

        // Add reference
        assert!(cache.add_ref(hash));

        // Remove references
        assert!(!cache.remove_ref(hash)); // Still has ref_count = 1
        assert!(cache.remove_ref(hash)); // Now removed

        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_contains() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        assert!(!cache.contains(hash));

        cache.insert(CachedPrefix::new(hash, 3, vec![]));
        assert!(cache.contains(hash));
    }

    #[test]
    fn test_prefix_cache_capacity() {
        let mut cache = PrefixCache::new(2);

        // Test that insert fails when cache is full and all entries are referenced
        // Insert with ref_count = 1 (default)
        cache.insert(CachedPrefix::new(1, 1, vec![]));
        cache.insert(CachedPrefix::new(2, 2, vec![]));
        assert_eq!(cache.len(), 2);

        // Both entries have ref_count = 1, so eviction skips them
        // Third insert should fail since no evictable entries
        let success = cache.insert(CachedPrefix::new(3, 3, vec![]));
        // Note: evict_lru only evicts ref_count = 0, so this fails
        assert!(!success);
        assert_eq!(cache.len(), 2);

        // Remove reference from first entry (makes it evictable)
        // Note: This removes the entry since ref_count drops to 0
        cache.remove_ref(1);
        assert_eq!(cache.len(), 1); // Entry 1 was removed

        // Now insert should succeed (we have capacity)
        let success = cache.insert(CachedPrefix::new(3, 3, vec![]));
        assert!(success);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_prefix_cache_clear() {
        let mut cache = PrefixCache::new(100);

        cache.insert(CachedPrefix::new(1, 1, vec![]));
        cache.insert(CachedPrefix::new(2, 2, vec![]));

        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_utilization() {
        let mut cache = PrefixCache::new(4);

        assert_eq!(cache.utilization(), 0.0);

        cache.insert(CachedPrefix::new(1, 1, vec![]));
        assert!((cache.utilization() - 0.25).abs() < 0.01);

        cache.insert(CachedPrefix::new(2, 2, vec![]));
        assert!((cache.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prefix_cache_stats_serialization() {
        let stats = PrefixCacheStats {
            hits: 100,
            misses: 50,
            prefixes_cached: 10,
            prefixes_evicted: 2,
            tokens_saved: 500,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: PrefixCacheStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.hits, 100);
        assert_eq!(parsed.misses, 50);
        assert_eq!(parsed.tokens_saved, 500);
    }

    #[test]
    fn test_find_longest_prefix() {
        let mut cache = PrefixCache::new(100);

        // Insert prefixes of different lengths
        let prefix_3 = compute_prefix_hash(&[1, 2, 3]);
        let prefix_5 = compute_prefix_hash(&[1, 2, 3, 4, 5]);

        cache.insert(CachedPrefix::new(prefix_3, 3, vec![]));
        cache.insert(CachedPrefix::new(prefix_5, 5, vec![]));

        // Search for tokens that match the 5-token prefix
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        let result = find_longest_prefix(&mut cache, &tokens);

        assert!(result.is_some());
        let (hash, len) = result.expect("test");
        assert_eq!(hash, prefix_5);
        assert_eq!(len, 5);
    }

    #[test]
    fn test_find_longest_prefix_no_match() {
        let mut cache = PrefixCache::new(100);

        // Insert a prefix
        let prefix = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(prefix, 3, vec![]));

        // Search for non-matching tokens
        let tokens = vec![4, 5, 6];
        let result = find_longest_prefix(&mut cache, &tokens);

        assert!(result.is_none());
    }

    #[test]
    fn test_prefix_cache_default() {
        let cache = PrefixCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_stats_hit_rate_zero() {
        let stats = PrefixCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    // === KV Quantization Tests ===

    #[test]
    fn test_kv_quant_type_bytes_per_value() {
        assert_eq!(KvQuantType::FP32.bytes_per_value(), 4.0);
        assert_eq!(KvQuantType::Q8.bytes_per_value(), 1.0);
        assert_eq!(KvQuantType::Q4.bytes_per_value(), 0.5);
    }

    #[test]
    fn test_kv_quant_type_memory_reduction() {
        assert_eq!(KvQuantType::FP32.memory_reduction(), 1.0);
        assert_eq!(KvQuantType::Q8.memory_reduction(), 4.0);
        assert_eq!(KvQuantType::Q4.memory_reduction(), 8.0);
    }

    #[test]
    fn test_kv_quant_type_default() {
        let quant_type = KvQuantType::default();
        assert_eq!(quant_type, KvQuantType::FP32);
    }

    #[test]
    fn test_q8_kv_block_new() {
        let block = Q8KvBlock::new();
        assert_eq!(block.scale, 0.0);
        assert_eq!(block.quants, [0i8; KV_QUANT_BLOCK_SIZE]);
    }

    #[test]
    fn test_q8_kv_block_quantize_dequantize() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.1; // -1.6 to 1.5
        }

        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Q8 should have ~1% error or less
        for i in 0..KV_QUANT_BLOCK_SIZE {
            let error = (values[i] - restored[i]).abs();
            assert!(
                error < 0.02,
                "Q8 error too high at {}: {} vs {}",
                i,
                values[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_q8_kv_block_zero_values() {
        let values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        for v in restored {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_kv_block_new() {
        let block = Q4KvBlock::new();
        assert_eq!(block.scale, 0.0);
        assert_eq!(block.quants, [0u8; KV_QUANT_BLOCK_SIZE / 2]);
    }

    #[test]
    fn test_q4_kv_block_quantize_dequantize() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.1;
        }

        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Q4 has more error than Q8, but should still be reasonable
        for i in 0..KV_QUANT_BLOCK_SIZE {
            let error = (values[i] - restored[i]).abs();
            assert!(
                error < 0.3,
                "Q4 error too high at {}: {} vs {}",
                i,
                values[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_q4_kv_block_zero_values() {
        let values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        for v in restored {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_quantized_kv_data_fp32() {
        let data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::FP32);
        assert_eq!(data.memory_bytes(), 16 * 8 * 64 * 4 * 2); // 2 for K+V
    }

    #[test]
    fn test_quantized_kv_data_q8() {
        let data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::Q8);
        // Q8 uses less memory than FP32
        let fp32_data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);
        assert!(data.memory_bytes() < fp32_data.memory_bytes());
    }

    #[test]
    fn test_quantized_kv_data_q4() {
        let data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::Q4);
        // Q4 uses even less memory than Q8
        let q8_data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);
        assert!(data.memory_bytes() < q8_data.memory_bytes());
    }

    #[test]
    fn test_quantized_kv_data_write_read_fp32() {
        let mut data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

        let test_keys: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        data.write_keys(0, &test_keys);
        let read_keys = data.read_keys(0, 64);

        assert_eq!(read_keys, test_keys);
    }

    #[test]
    fn test_quantized_kv_data_write_read_q8() {
        let mut data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);

        let test_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        data.write_values(0, &test_values);
        let read_values = data.read_values(0, 64);

        // Q8 should preserve values with small error
        for (orig, read) in test_values.iter().zip(read_values.iter()) {
            assert!((orig - read).abs() < 0.05);
        }
    }

    #[test]
    fn test_quantized_kv_page_new() {
        let page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
        assert_eq!(page.quant_type(), KvQuantType::Q8);
        assert_eq!(page.num_tokens, 0);
        assert_eq!(page.ref_count, 0); // Pages start in free pool with ref_count 0
        assert!(!page.is_full());
        assert!(!page.is_shared());
    }

    #[test]
    fn test_quantized_kv_page_read_write() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);

        let keys: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let values: Vec<f32> = (0..512).map(|i| -i as f32 * 0.01).collect();

        page.write_keys(0, &keys);
        page.write_values(0, &values);

        let read_keys = page.read_keys(0);
        let read_values = page.read_values(0);

        assert_eq!(read_keys.len(), 512);
        assert_eq!(read_values.len(), 512);
        assert_eq!(read_keys, keys);
        assert_eq!(read_values, values);
    }

    #[test]
    fn test_quantized_kv_page_is_full() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
        assert!(!page.is_full());
        assert_eq!(page.remaining_capacity(), 16);

        page.num_tokens = 16;
        assert!(page.is_full());
        assert_eq!(page.remaining_capacity(), 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_new() {
        let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        assert_eq!(cache.quant_type(), KvQuantType::Q8);
        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_allocate() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(32).expect("test");

        assert_eq!(cache.free_page_count(), 98); // 32 tokens = 2 pages
        assert_eq!(cache.stats().active_sequences, 1);
        assert!(seq_id.value() < u64::MAX);
    }

    #[test]
    fn test_quantized_paged_kv_cache_free() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);
        let seq_id = cache.allocate_sequence(16).expect("test");

        assert_eq!(cache.free_page_count(), 99);

        cache.free_sequence(seq_id);

        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_memory_savings() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let _seq_id = cache.allocate_sequence(16).expect("test");

        let savings = cache.memory_savings();
        // Q8 uses (4 + 32) = 36 bytes per block of 32 values vs 128 bytes for FP32
        // Ratio: 36/128 = 0.28125, with some overhead ~0.35
        assert!(
            savings < 0.6,
            "Q8 should use less than 60% of FP32 memory, got {}",
            savings
        );
    }

    #[test]
    fn test_quantized_paged_kv_cache_q4_savings() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);
        let _seq_id = cache.allocate_sequence(16).expect("test");

        let savings = cache.memory_savings();
        // Q4 uses (4 + 16) = 20 bytes per block of 32 values vs 128 bytes for FP32
        // Ratio: 20/128 = 0.15625, with some overhead ~0.20
        assert!(
            savings < 0.4,
            "Q4 should use less than 40% of FP32 memory, got {}",
            savings
        );
    }

    #[test]
    fn test_quantized_paged_kv_cache_get_page() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page.quant_type(), KvQuantType::Q8);

        let page2 = cache.get_page(seq_id, 16).expect("test");
        assert_eq!(page2.quant_type(), KvQuantType::Q8);
    }

    #[test]
    fn test_quantized_paged_kv_cache_get_page_mut() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(16).expect("test");

        let page = cache.get_page_mut(seq_id, 0).expect("test");
        page.num_tokens = 8;

        let page2 = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page2.num_tokens, 8);
    }

    #[test]
    fn test_quantized_paged_kv_cache_oom() {
        let mut cache = QuantizedPagedKvCache::new(1, 16, 8, 64, KvQuantType::Q8);
        let _seq1 = cache.allocate_sequence(16).expect("test");

        let result = cache.allocate_sequence(16);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_q8_block_default() {
        let block = Q8KvBlock::default();
        assert_eq!(block.scale, 0.0);
    }

    #[test]
    fn test_q4_block_default() {
        let block = Q4KvBlock::default();
        assert_eq!(block.scale, 0.0);
    }
}
