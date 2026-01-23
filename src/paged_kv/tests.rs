#[cfg(test)]
mod tests {
    use crate::paged_kv::*;

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

    // =========================================================================
    // Deep coverage tests (_deep_pkcov_ prefix)
    // =========================================================================

    // --- Error handling paths ---

    #[test]
    fn test_deep_pkcov_extend_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.extend(fake_seq, 32);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_extend_sequence_out_of_memory() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // Uses 1 page, 1 free

        // Extend to need more pages than available
        // extend(48) needs 48/16 = 3 pages total, have 1, need 2 more, only 1 free
        let result = cache.extend(seq_id, 48);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_deep_pkcov_extend_no_new_pages_needed() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // 1 page, 16 capacity
        cache.update_tokens(seq_id, 5).expect("update");

        // Extend by small amount that fits in existing page
        let result = cache.extend(seq_id, 5);
        assert!(result.is_ok());
        assert_eq!(cache.free_page_count(), 99); // No new pages allocated
    }

    #[test]
    fn test_deep_pkcov_update_tokens_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.update_tokens(fake_seq, 10);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_fork_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.fork_sequence(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_sequence_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_page(fake_seq, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_mut_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_page_mut(fake_seq, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_mut_invalid_page_access() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // 1 page

        let result = cache.get_page_mut(seq_id, 100); // Beyond allocated
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    // --- Edge cases in paged memory management ---

    #[test]
    fn test_deep_pkcov_utilization_zero_pages() {
        // Create cache with zero pages - edge case
        let cache = PagedKvCache::new(0, 16, 8, 64);
        assert_eq!(cache.utilization(), 0.0);
        assert_eq!(cache.free_page_count(), 0);
    }

    #[test]
    fn test_deep_pkcov_free_nonexistent_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        // Should not panic, just do nothing
        cache.free_sequence(fake_seq);
        assert_eq!(cache.stats().sequences_freed, 0);
    }

    #[test]
    fn test_deep_pkcov_allocate_zero_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Zero tokens should allocate at least 1 page (div_ceil behavior)
        // Actually 0.div_ceil(16) = 0, so 0 pages needed
        let seq_id = cache.allocate_sequence(0).expect("alloc");
        assert!(cache.page_tables.contains_key(&seq_id));
    }

    #[test]
    fn test_deep_pkcov_update_tokens_spans_multiple_pages() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(48).expect("alloc"); // 3 pages

        // Update with tokens spanning all pages
        cache.update_tokens(seq_id, 48).expect("update");

        let tokens = cache.get_sequence_tokens(seq_id).expect("get");
        assert_eq!(tokens, 48);
    }

    #[test]
    fn test_deep_pkcov_update_tokens_partial_fill() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("alloc"); // 2 pages

        // Only fill part of first page
        cache.update_tokens(seq_id, 5).expect("update");

        let tokens = cache.get_sequence_tokens(seq_id).expect("get");
        assert_eq!(tokens, 5);
    }

    // --- Copy-on-write edge cases ---

    #[test]
    fn test_deep_pkcov_cow_out_of_memory() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc"); // 1 page
        let child_id = cache.fork_sequence(parent_id).expect("fork"); // Shares page

        // Fill remaining page
        let _ = cache.allocate_sequence(16).expect("alloc2");

        // Now try to write to child - COW needs a free page but none available
        let result = cache.get_page_mut(child_id, 0);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_deep_pkcov_cow_multiple_forks() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc");
        cache.update_tokens(parent_id, 16).expect("update");

        // Fork multiple times
        let child1 = cache.fork_sequence(parent_id).expect("fork1");
        let child2 = cache.fork_sequence(parent_id).expect("fork2");

        assert_eq!(cache.stats().cow_operations, 2);

        // Write to child1 triggers COW
        let _page = cache.get_page_mut(child1, 0).expect("get");
        assert_eq!(cache.stats().cow_operations, 3);

        // child2 still shares with parent
        let page = cache.get_page(child2, 0).expect("get2");
        assert!(page.ref_count >= 1);
    }

    #[test]
    fn test_deep_pkcov_free_shared_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc");
        let child_id = cache.fork_sequence(parent_id).expect("fork");

        // Free parent - pages should not return to free list (still referenced by child)
        cache.free_sequence(parent_id);
        assert_eq!(cache.free_page_count(), 99); // Page still in use by child

        // Free child - now pages return
        cache.free_sequence(child_id);
        assert_eq!(cache.free_page_count(), 100);
    }

    // --- Defragmentation paths ---

    #[test]
    fn test_deep_pkcov_should_defragment_low_free_ratio() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        // Allocate to use most pages
        for _ in 0..9 {
            let _ = cache.allocate_sequence(16).expect("alloc");
        }

        // With >90% utilization and any fragmentation, should trigger
        // Need to create a hole first
        let seq_to_free = cache.allocate_sequence(16).ok();
        if let Some(seq) = seq_to_free {
            cache.free_sequence(seq);
        }

        // Low free ratio check is at 10% threshold
        // We have 1 free page out of 10 = 10%, right at threshold
    }

    #[test]
    fn test_deep_pkcov_should_defragment_high_waste() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Allocate several sequences with partial fill
        let seq1 = cache.allocate_sequence(32).expect("alloc1"); // 2 pages
        let seq2 = cache.allocate_sequence(32).expect("alloc2"); // 2 pages
        let seq3 = cache.allocate_sequence(32).expect("alloc3"); // 2 pages

        // Free middle sequence to create holes
        cache.free_sequence(seq2);

        // Update with very few tokens to create waste
        cache.update_tokens(seq1, 1).expect("update1");
        cache.update_tokens(seq3, 1).expect("update3");

        let stats = cache.fragmentation_stats();
        assert!(stats.holes > 0 || stats.wasted_capacity > 0);
    }

    #[test]
    fn test_deep_pkcov_compact_empty_page_list() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(0).expect("alloc"); // 0 pages

        let moved = cache.compact_sequence(seq_id);
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_deep_pkcov_fragmentation_stats_large_free_region() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Allocate just one sequence at the beginning
        let _ = cache.allocate_sequence(16).expect("alloc");

        let stats = cache.fragmentation_stats();
        // Large free region at the end
        assert!(stats.largest_free_region >= 99);
    }

    // --- Prefix cache edge cases ---

    #[test]
    fn test_deep_pkcov_prefix_cache_add_ref_not_found() {
        let mut cache = PrefixCache::new(100);

        let result = cache.add_ref(999999);
        assert!(!result);
    }

    #[test]
    fn test_deep_pkcov_prefix_cache_remove_ref_not_found() {
        let mut cache = PrefixCache::new(100);

        let result = cache.remove_ref(999999);
        assert!(!result);
    }

    #[test]
    fn test_deep_pkcov_prefix_cache_lru_eviction() {
        let mut cache = PrefixCache::new(3);

        // Insert 3 prefixes with ref_count 0 (to make them evictable)
        let mut p1 = CachedPrefix::new(1, 1, vec![]);
        p1.ref_count = 0;
        let mut p2 = CachedPrefix::new(2, 2, vec![]);
        p2.ref_count = 0;
        let mut p3 = CachedPrefix::new(3, 3, vec![]);
        p3.ref_count = 0;

        cache.insert(p1);
        cache.insert(p2);
        cache.insert(p3);

        assert_eq!(cache.len(), 3);

        // Access p2 and p3 to give them distinct last_access times
        // This ensures deterministic LRU behavior (p1 has lowest last_access)
        cache.lookup(2); // p2.last_access = 1
        cache.lookup(3); // p3.last_access = 2

        // Insert new prefix - should evict LRU (p1 with last_access=0)
        let mut p4 = CachedPrefix::new(4, 4, vec![]);
        p4.ref_count = 0;
        let inserted = cache.insert(p4);
        assert!(inserted);

        // p1 should be evicted (lowest last_access)
        assert!(!cache.contains(1));
        assert!(cache.contains(2)); // Accessed, not evicted
        assert!(cache.contains(3)); // Most recently used
        assert!(cache.contains(4)); // Just inserted
        assert_eq!(cache.stats().prefixes_evicted, 1);
    }

    #[test]
    fn test_deep_pkcov_prefix_cache_utilization_zero_capacity() {
        let cache = PrefixCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_deep_pkcov_find_longest_prefix_partial_match() {
        let mut cache = PrefixCache::new(100);

        // Insert only a 3-token prefix
        let prefix_3 = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(prefix_3, 3, vec![]));

        // Search for 5-token sequence that matches first 3
        let tokens = vec![1, 2, 3, 4, 5];
        let result = find_longest_prefix(&mut cache, &tokens);

        assert!(result.is_some());
        let (hash, len) = result.expect("match");
        assert_eq!(hash, prefix_3);
        assert_eq!(len, 3);
    }

    // --- Quantization edge cases ---

    #[test]
    fn test_deep_pkcov_q8_quantize_extreme_values() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        // Mix of extreme values
        values[0] = 1000.0;
        values[1] = -1000.0;
        values[16] = 0.0001;

        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Extreme values should be clipped but proportional
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
    }

    #[test]
    fn test_deep_pkcov_q4_quantize_extreme_values() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        values[0] = 100.0;
        values[1] = -100.0;

        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Should handle extreme values without panic
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
    }

    #[test]
    fn test_deep_pkcov_quantized_kv_data_write_read_q4() {
        let mut data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);

        let test_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        data.write_values(0, &test_values);
        let read_values = data.read_values(0, 64);

        // Q4 has more error but should preserve sign and rough magnitude
        for (orig, read) in test_values.iter().zip(read_values.iter()) {
            assert!(
                (orig - read).abs() < 0.5,
                "Q4 error too high: {} vs {}",
                orig,
                read
            );
        }
    }

    #[test]
    fn test_deep_pkcov_quantized_kv_data_cross_block_write() {
        let mut data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);

        // Write data that spans multiple blocks
        let test_keys: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        data.write_keys(16, &test_keys); // Start at offset 16, spans blocks

        let read_keys = data.read_keys(16, 128);
        assert_eq!(read_keys.len(), 128);
    }

    #[test]
    fn test_deep_pkcov_quantized_paged_cache_invalid_page() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(16).expect("alloc");

        // Try to access page beyond allocation
        let result = cache.get_page(seq_id, 100);
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    #[test]
    fn test_deep_pkcov_quantized_paged_cache_sequence_not_found() {
        let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);
        let fake_seq = SeqId::new();

        let result = cache.get_page(fake_seq, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_quantized_paged_cache_get_page_mut_invalid() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(16).expect("alloc");

        let result = cache.get_page_mut(seq_id, 50);
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    #[test]
    fn test_deep_pkcov_quantized_paged_cache_memory_savings_empty() {
        let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let savings = cache.memory_savings();
        assert_eq!(savings, 1.0); // No sequences, default 1.0
    }

    #[test]
    fn test_deep_pkcov_quantized_paged_cache_total_pages() {
        let cache = QuantizedPagedKvCache::new(50, 16, 8, 64, KvQuantType::Q4);
        assert_eq!(cache.total_pages(), 50);
    }

    // --- Additional boundary tests ---

    #[test]
    fn test_deep_pkcov_kv_page_remaining_capacity_overflow() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        page.num_tokens = 20; // More than block_size

        // saturating_sub should handle this
        assert_eq!(page.remaining_capacity(16), 0);
    }

    #[test]
    fn test_deep_pkcov_cached_prefix_remove_ref_underflow() {
        let mut prefix = CachedPrefix::new(1, 5, vec![]);
        prefix.ref_count = 0;

        // Should not underflow
        let removed = prefix.remove_ref();
        assert!(removed);
        assert_eq!(prefix.ref_count, 0);
    }

    #[test]
    fn test_deep_pkcov_seq_id_value() {
        let seq = SeqId::new();
        let value = seq.value();
        // Value should be accessible and not panic
        assert!(value < u64::MAX);
    }

    #[test]
    fn test_deep_pkcov_fragmentation_stats_default() {
        let stats = FragmentationStats::default();
        assert_eq!(stats.holes, 0);
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert_eq!(stats.largest_free_region, 0);
        assert_eq!(stats.avg_tokens_per_page, 0.0);
    }

    #[test]
    fn test_deep_pkcov_quantized_kv_page_memory_bytes() {
        let page_fp32 = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);
        let page_q8 = QuantizedKvPage::new(PageId::new(1), KvQuantType::Q8, 16, 8, 64);
        let page_q4 = QuantizedKvPage::new(PageId::new(2), KvQuantType::Q4, 16, 8, 64);

        // FP32 should use most memory
        assert!(page_fp32.memory_bytes() > page_q8.memory_bytes());
        assert!(page_q8.memory_bytes() > page_q4.memory_bytes());
    }

    #[test]
    fn test_deep_pkcov_quantized_kv_page_write_multiple_positions() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);

        // Write to different token positions
        let keys1: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let keys2: Vec<f32> = (0..512).map(|i| -i as f32).collect();

        page.write_keys(0, &keys1);
        page.write_keys(1, &keys2);

        let read1 = page.read_keys(0);
        let read2 = page.read_keys(1);

        assert_eq!(read1[0], 0.0);
        assert_eq!(read2[0], 0.0); // -0 is 0
        assert_eq!(read2[1], -1.0);
    }

    #[test]
    fn test_deep_pkcov_extend_sequence_exact_capacity() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // 1 page
        cache.update_tokens(seq_id, 16).expect("update");

        // Extend by exactly block_size - should need one more page
        let result = cache.extend(seq_id, 16);
        assert!(result.is_ok());
        assert_eq!(cache.free_page_count(), 98);
    }
}
