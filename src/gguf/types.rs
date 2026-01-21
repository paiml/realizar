//! GGUF Types and Constants
//!
//! Core constants for GGUF format parsing.
//! Note: GGUFValue enum remains in monolith during migration.

// Re-export from monolith - these will be migrated here incrementally
// For now, this module only adds tests for the constants

#[cfg(test)]
mod tests {
    use crate::gguf::{
        ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE, GGUF_MAGIC,
        GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
        GGUF_TYPE_Q8_0, GGUF_VERSION_V3, HIDDEN_BUFFER_INLINE_CAP, TOKEN_BUFFER_INLINE_CAP,
    };

    #[test]
    fn test_magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
    }

    #[test]
    fn test_quantization_constants() {
        assert_eq!(GGUF_TYPE_F32, 0);
        assert_eq!(GGUF_TYPE_F16, 1);
        assert_eq!(GGUF_TYPE_Q4_0, 2);
        assert_eq!(GGUF_TYPE_Q8_0, 8);
        assert_eq!(GGUF_TYPE_Q4_K, 12);
        assert_eq!(GGUF_TYPE_Q6_K, 14);
    }

    #[test]
    fn test_buffer_constants() {
        assert_eq!(TOKEN_BUFFER_INLINE_CAP, 32);
        assert_eq!(ATTENTION_BUFFER_INLINE_CAP, 64);
        assert_eq!(HIDDEN_BUFFER_INLINE_CAP, 128);
    }

    #[test]
    fn test_buffer_watermarks() {
        assert_eq!(BUFFER_LW_SIZE, 1024);
        assert_eq!(BUFFER_HW_SIZE, 8 * 1024);
        assert_eq!(BUFFER_MAX_SIZE, 32 * 1024);
    }

    #[test]
    fn test_version_constant() {
        assert_eq!(GGUF_VERSION_V3, 3);
    }
}

/// GPU-specific tests for batching infrastructure (PARITY-023)
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use crate::gguf::{
        BatchRequestCollector, BatchingConfig, ContinuousBatchScheduler, GpuBufferPool,
        PendingRequest, RequestBatch, SlotState, SpeculativeConfig, SpeculativeDecoder,
        VerificationResult,
    };

    // =========================================================================
    // PARITY-023: Request Batching Tests
    // =========================================================================

    #[test]
    fn test_pending_request_creation() {
        let req = PendingRequest::new(1, vec![1, 2, 3], 10, 0.7, 40);
        assert_eq!(req.id, 1);
        assert_eq!(req.prompt, vec![1, 2, 3]);
        assert_eq!(req.max_tokens, 10);
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert_eq!(req.top_k, 40);
    }

    #[test]
    fn test_pending_request_wait_time() {
        let req = PendingRequest::new(1, vec![1], 10, 0.7, 40);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(req.wait_time().as_millis() >= 10);
    }

    #[test]
    fn test_request_batch_creation() {
        let reqs = vec![
            PendingRequest::new(1, vec![1, 2], 10, 0.7, 40),
            PendingRequest::new(2, vec![3, 4], 20, 0.8, 50),
        ];
        let batch = RequestBatch::new(reqs);
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.prompts().len(), 2);
    }

    #[test]
    fn test_request_batch_avg_wait_time() {
        let reqs = vec![
            PendingRequest::new(1, vec![1], 10, 0.7, 40),
            PendingRequest::new(2, vec![2], 10, 0.7, 40),
        ];
        let batch = RequestBatch::new(reqs);
        // Should return some duration (not panic)
        let _ = batch.avg_wait_time();
    }

    #[test]
    fn test_request_batch_empty_avg_wait() {
        let batch = RequestBatch::new(vec![]);
        assert_eq!(batch.avg_wait_time(), std::time::Duration::ZERO);
    }

    #[test]
    fn test_batch_collector_creation() {
        let collector = BatchRequestCollector::new();
        assert_eq!(collector.batch_threshold, 32);
        assert_eq!(collector.timeout_ms, 50);
        assert_eq!(collector.max_batch_size, 64);
    }

    #[test]
    fn test_batch_collector_custom_thresholds() {
        let collector = BatchRequestCollector::with_thresholds(8, 100, 32);
        assert_eq!(collector.batch_threshold, 8);
        assert_eq!(collector.timeout_ms, 100);
        assert_eq!(collector.max_batch_size, 32);
    }

    #[test]
    fn test_batch_collector_submit() {
        let collector = BatchRequestCollector::new();
        let id1 = collector.submit(vec![1, 2, 3], 10, 0.7, 40);
        let id2 = collector.submit(vec![4, 5, 6], 20, 0.8, 50);
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(collector.pending_count(), 2);
        assert_eq!(collector.total_submitted(), 2);
    }

    #[test]
    fn test_batch_collector_not_ready_below_threshold() {
        let collector = BatchRequestCollector::with_thresholds(32, 1000, 64);
        collector.submit(vec![1], 10, 0.7, 40);
        assert!(!collector.is_batch_ready());
    }

    #[test]
    fn test_batch_collector_ready_at_threshold() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 64);
        collector.submit(vec![1], 10, 0.7, 40);
        collector.submit(vec![2], 10, 0.7, 40);
        assert!(collector.is_batch_ready());
    }

    #[test]
    fn test_batch_collector_collect_batch() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 64);
        collector.submit(vec![1], 10, 0.7, 40);
        collector.submit(vec![2], 10, 0.7, 40);
        let batch = collector.collect_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().size(), 2);
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_batch_collector_flush() {
        let collector = BatchRequestCollector::new();
        collector.submit(vec![1], 10, 0.7, 40);
        let batch = collector.flush();
        assert!(batch.is_some());
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_batch_collector_flush_empty() {
        let collector = BatchRequestCollector::new();
        assert!(collector.flush().is_none());
    }

    #[test]
    fn test_batch_collector_default() {
        let collector = BatchRequestCollector::default();
        assert_eq!(collector.batch_threshold, 32);
    }

    // =========================================================================
    // PARITY-023: BatchingConfig Tests
    // =========================================================================

    #[test]
    fn test_batching_config_default() {
        let config = BatchingConfig::default();
        assert_eq!(config.batch_threshold, 32);
        assert_eq!(config.timeout_ms, 50);
        assert_eq!(config.max_batch_size, 64);
        assert!(config.prefer_throughput);
    }

    #[test]
    fn test_batching_config_latency_optimized() {
        let config = BatchingConfig::latency_optimized();
        assert_eq!(config.batch_threshold, 8);
        assert_eq!(config.timeout_ms, 10);
        assert!(!config.prefer_throughput);
    }

    #[test]
    fn test_batching_config_throughput_optimized() {
        let config = BatchingConfig::throughput_optimized();
        assert_eq!(config.batch_threshold, 32);
        assert_eq!(config.timeout_ms, 100);
        assert!(config.prefer_throughput);
    }

    // =========================================================================
    // PARITY-028: SlotState Tests
    // =========================================================================

    #[test]
    fn test_slot_state_empty() {
        let slot = SlotState::Empty;
        assert!(slot.is_empty());
        assert!(!slot.is_active());
        assert!(!slot.is_completed());
        assert!(slot.request_id().is_none());
    }

    #[test]
    fn test_slot_state_active() {
        let slot = SlotState::Active {
            request_id: 42,
            prompt_tokens: vec![1, 2, 3],
            generated_tokens: vec![4, 5],
            max_tokens: 100,
            temperature: 0.7,
            top_k: 40,
        };
        assert!(!slot.is_empty());
        assert!(slot.is_active());
        assert!(!slot.is_completed());
        assert_eq!(slot.request_id(), Some(42));
    }

    #[test]
    fn test_slot_state_completed() {
        let slot = SlotState::Completed {
            request_id: 42,
            generated_tokens: vec![1, 2, 3, 4, 5],
        };
        assert!(!slot.is_empty());
        assert!(!slot.is_active());
        assert!(slot.is_completed());
        assert_eq!(slot.request_id(), Some(42));
    }

    // =========================================================================
    // PARITY-028: ContinuousBatchScheduler Tests
    // =========================================================================

    #[test]
    fn test_continuous_batch_scheduler_creation() {
        let scheduler = ContinuousBatchScheduler::new(4, 32, 2560, 2048);
        assert_eq!(scheduler.num_slots, 4);
        assert_eq!(scheduler.empty_count(), 4);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_continuous_batch_scheduler_submit() {
        let scheduler = ContinuousBatchScheduler::new(4, 32, 2560, 2048);
        let id = scheduler.submit(vec![1, 2, 3], 100, 0.7, 40);
        assert!(id.is_some());
        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(scheduler.empty_count(), 3);
    }

    #[test]
    fn test_continuous_batch_scheduler_full() {
        let scheduler = ContinuousBatchScheduler::new(2, 32, 2560, 2048);
        scheduler.submit(vec![1], 100, 0.7, 40);
        scheduler.submit(vec![2], 100, 0.7, 40);
        let id = scheduler.submit(vec![3], 100, 0.7, 40);
        assert!(id.is_none()); // No slots available
    }

    #[test]
    fn test_continuous_batch_scheduler_utilization() {
        let scheduler = ContinuousBatchScheduler::new(4, 32, 2560, 2048);
        assert!((scheduler.utilization() - 0.0).abs() < 0.01);
        scheduler.submit(vec![1], 100, 0.7, 40);
        assert!((scheduler.utilization() - 0.25).abs() < 0.01);
        scheduler.submit(vec![2], 100, 0.7, 40);
        assert!((scheduler.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_continuous_batch_scheduler_complete_request() {
        let scheduler = ContinuousBatchScheduler::new(4, 32, 2560, 2048);
        scheduler.submit(vec![1], 100, 0.7, 40);
        assert!(!scheduler.has_completed());
        scheduler.complete_request(0, vec![10, 20, 30]);
        assert!(scheduler.has_completed());
        let completed = scheduler.poll_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].1, vec![10, 20, 30]);
    }

    #[test]
    fn test_continuous_batch_scheduler_get_active_slots() {
        let scheduler = ContinuousBatchScheduler::new(4, 32, 2560, 2048);
        scheduler.submit(vec![1, 2, 3], 100, 0.7, 40);
        let active = scheduler.get_active_slots();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, 0); // slot index
        assert_eq!(active[0].1, 3); // position (prompt length)
    }

    // =========================================================================
    // PARITY-029: SpeculativeDecoder Tests
    // =========================================================================

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.speculation_length, 4);
        assert!((config.draft_temperature - 0.0).abs() < 0.01);
        assert!(config.self_speculative);
    }

    #[test]
    fn test_speculative_decoder_creation() {
        let decoder = SpeculativeDecoder::new();
        assert!((decoder.acceptance_rate() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_with_config() {
        let config = SpeculativeConfig {
            speculation_length: 8,
            draft_temperature: 0.5,
            self_speculative: false,
        };
        let decoder = SpeculativeDecoder::with_config(config);
        assert_eq!(decoder.config.speculation_length, 8);
    }

    #[test]
    fn test_speculative_decoder_verify_draft_greedy() {
        let decoder = SpeculativeDecoder::new();
        let draft_tokens = vec![5, 3, 7];
        let target_logits = vec![
            {
                let mut l = vec![0.0; 10];
                l[5] = 1.0; // Matches draft
                l
            },
            {
                let mut l = vec![0.0; 10];
                l[3] = 1.0; // Matches draft
                l
            },
            {
                let mut l = vec![0.0; 10];
                l[8] = 1.0; // Different from draft
                l
            },
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);
        assert_eq!(result.draft_count, 3);
        // First two match, third diverges but we get target's token
        assert!(result.accepted_count >= 2);
    }

    #[test]
    fn test_speculative_decoder_acceptance_rate() {
        let decoder = SpeculativeDecoder::new();
        let draft_tokens = vec![5];
        let target_logits = vec![{
            let mut l = vec![0.0; 10];
            l[5] = 1.0;
            l
        }];

        decoder.verify_draft(&draft_tokens, &target_logits, 0.0);
        assert!(decoder.acceptance_rate() > 0.0);
    }

    #[test]
    fn test_speculative_decoder_expected_speedup() {
        let decoder = SpeculativeDecoder::new();
        // With no drafts, speedup is 1.0
        assert!((decoder.expected_speedup() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_reset_stats() {
        let decoder = SpeculativeDecoder::new();
        let draft_tokens = vec![5];
        let target_logits = vec![{
            let mut l = vec![0.0; 10];
            l[5] = 1.0;
            l
        }];
        decoder.verify_draft(&draft_tokens, &target_logits, 0.0);
        decoder.reset_stats();
        assert!((decoder.acceptance_rate() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_default() {
        let decoder = SpeculativeDecoder::default();
        assert_eq!(decoder.config.speculation_length, 4);
    }

    // =========================================================================
    // PARITY-031: GpuBufferPool Tests
    // =========================================================================

    #[test]
    fn test_gpu_buffer_pool_creation() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        assert!(!pool.is_zero_alloc()); // Not warmed up yet
    }

    #[test]
    fn test_gpu_buffer_pool_warmup() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        pool.warmup();
        assert!(pool.is_zero_alloc());
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_hidden() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        pool.warmup();
        let buffer = pool.borrow_hidden();
        assert_eq!(buffer.len(), 256);
        pool.return_hidden(buffer);
        assert!(pool.is_zero_alloc());
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_intermediate() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        pool.warmup();
        let buffer = pool.borrow_intermediate();
        assert_eq!(buffer.len(), 512);
        pool.return_intermediate(buffer);
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_attention() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        pool.warmup();
        let buffer = pool.borrow_attention();
        assert_eq!(buffer.len(), 8 * 2048); // num_heads * max_seq_len
        pool.return_attention(buffer);
    }

    #[test]
    fn test_gpu_buffer_pool_stats() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        pool.warmup();
        let _ = pool.borrow_hidden();
        let stats = pool.stats();
        assert_eq!(stats.borrows, 1);
        assert!(stats.warmed_up);
    }

    #[test]
    fn test_gpu_buffer_pool_memory_usage() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 4);
        let expected = 4 * 256 * 4 + 4 * 512 * 4 + 4 * 8 * 2048 * 4;
        assert_eq!(pool.memory_usage_bytes(), expected);
    }

    #[test]
    fn test_gpu_buffer_pool_post_warmup_alloc() {
        let pool = GpuBufferPool::new(256, 512, 2048, 8, 1); // Pool size 1
        pool.warmup();

        // Borrow the only buffer
        let b1 = pool.borrow_hidden();
        // Borrow another - this should allocate
        let _b2 = pool.borrow_hidden();

        let stats = pool.stats();
        assert_eq!(stats.post_warmup_allocs, 1);
        assert!(!pool.is_zero_alloc());

        pool.return_hidden(b1);
    }
}
