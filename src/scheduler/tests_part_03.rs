//! Scheduler Tests Part 03 - Deep Coverage
//!
//! Tests for DynamicPriorityScheduler, BatchConfig builders, Deadline handling,
//! and preemption logic. Targets uncovered branches in scheduler/mod.rs.
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use crate::paged_kv::PagedKvCache;
    use crate::scheduler::*;
    use std::collections::HashMap;

    // ========================================================================
    // Deadline Tests
    // ========================================================================

    #[test]
    fn test_deadline_default() {
        let deadline = Deadline::default();
        assert_eq!(deadline.target_latency_ms, 1000);
        assert!(deadline.hard_deadline_ms.is_none());
        assert!((deadline.sla_target - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_deadline_with_target() {
        let deadline = Deadline::with_target(500);
        assert_eq!(deadline.target_latency_ms, 500);
        assert!(deadline.hard_deadline_ms.is_none());
    }

    #[test]
    fn test_deadline_strict() {
        let deadline = Deadline::strict(200, 1000);
        assert_eq!(deadline.target_latency_ms, 200);
        assert_eq!(deadline.hard_deadline_ms, Some(1000));
        assert!((deadline.sla_target - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // DynamicPriorityConfig Tests
    // ========================================================================

    #[test]
    fn test_dynamic_priority_config_default() {
        let config = DynamicPriorityConfig::default();
        assert!(config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 5000);
        assert_eq!(config.max_promoted_priority, Priority::High);
        assert!(config.enable_deadline_scheduling);
        assert!((config.urgency_factor - 2.0).abs() < 0.001);
        assert_eq!(config.min_tokens_per_request, 1);
        assert!(config.enable_fair_share);
    }

    #[test]
    fn test_dynamic_priority_config_with_budgets() {
        let budgets = [0.1, 0.2, 0.3, 0.4];
        let config = DynamicPriorityConfig::with_budgets(budgets);
        assert_eq!(config.priority_budgets, budgets);
    }

    #[test]
    fn test_dynamic_priority_config_no_promotion() {
        let config = DynamicPriorityConfig::default().no_promotion();
        assert!(!config.enable_age_promotion);
    }

    #[test]
    fn test_dynamic_priority_config_with_promotion_interval() {
        let config = DynamicPriorityConfig::default().with_promotion_interval(10000);
        assert_eq!(config.promotion_interval_ms, 10000);
    }

    // ========================================================================
    // BatchConfig Tests
    // ========================================================================

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_ubatch_tokens, 512);
        assert_eq!(config.max_sbatch_sequences, 8);
        assert!(config.prefer_pure_decode);
        assert_eq!(config.max_context_length, 2048);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_batch_config_with_max_tokens() {
        let config = BatchConfig::default().with_max_tokens(1024);
        assert_eq!(config.max_ubatch_tokens, 1024);
    }

    #[test]
    fn test_batch_config_with_max_sequences() {
        let config = BatchConfig::default().with_max_sequences(16);
        assert_eq!(config.max_sbatch_sequences, 16);
    }

    // ========================================================================
    // DynamicRequest Tests
    // ========================================================================

    #[test]
    fn test_dynamic_request_new() {
        let request = DynamicRequest::new(42, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 42);
        assert_eq!(request.input_ids, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.original_priority, Priority::Normal);
        assert_eq!(request.effective_priority, Priority::Normal);
        assert!(request.deadline.is_none());
        assert_eq!(request.promotions, 0);
        assert_eq!(request.state, SequenceState::Waiting);
        assert!(request.generated_tokens.is_empty());
        assert!(request.seq_id.is_none());
        assert!(request.ttft_ms.is_none());
    }

    #[test]
    fn test_dynamic_request_with_priority() {
        let request = DynamicRequest::new(1, vec![1], 5).with_priority(Priority::Critical);
        assert_eq!(request.original_priority, Priority::Critical);
        assert_eq!(request.effective_priority, Priority::Critical);
    }

    #[test]
    fn test_dynamic_request_with_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(500));
        assert!(request.deadline.is_some());
        assert_eq!(request.deadline.as_ref().unwrap().target_latency_ms, 500);
    }

    #[test]
    fn test_dynamic_request_wait_time() {
        let request = DynamicRequest::new(1, vec![1], 5);
        // Wait time should be very small (just created)
        assert!(request.wait_time_ms() < 100);
    }

    #[test]
    fn test_dynamic_request_is_urgent_false() {
        let request =
            DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(10000));
        // Fresh request is not urgent
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_is_urgent_no_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5);
        // No deadline means never urgent
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_is_expired_no_hard_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(500));
        // No hard deadline, so never expired
        assert!(!request.is_expired());
    }

    #[test]
    fn test_dynamic_request_urgency_score_no_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5);
        assert!((request.urgency_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_urgency_score_zero_target() {
        let deadline = Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: None,
            sla_target: 0.99,
        };
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(deadline);
        assert!((request.urgency_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_remaining_tokens() {
        let mut request = DynamicRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![4, 5, 6];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_dynamic_request_remaining_tokens_overflow() {
        let mut request = DynamicRequest::new(1, vec![1], 5);
        request.generated_tokens = vec![1, 2, 3, 4, 5, 6, 7]; // More than max
        assert_eq!(request.remaining_tokens(), 0); // saturating_sub protects
    }

    #[test]
    fn test_dynamic_request_total_tokens() {
        let mut request = DynamicRequest::new(1, vec![1, 2, 3], 10);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    // ========================================================================
    // DynamicSchedulerStats Tests
    // ========================================================================

    #[test]
    fn test_dynamic_scheduler_stats_default() {
        let stats = DynamicSchedulerStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.sla_met, 0);
        assert_eq!(stats.sla_missed, 0);
        assert_eq!(stats.dropped_requests, 0);
        assert_eq!(stats.promotions, 0);
        assert!((stats.avg_ttft_ms - 0.0).abs() < 0.001);
        assert!((stats.p99_ttft_ms - 0.0).abs() < 0.001);
        assert_eq!(stats.tokens_by_priority, [0, 0, 0, 0]);
        assert_eq!(stats.queue_depth_by_priority, [0, 0, 0, 0]);
    }

    // ========================================================================
    // DynamicPriorityScheduler Full Coverage Tests
    // ========================================================================

    #[test]
    fn test_dynamic_scheduler_with_config() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let scheduler = DynamicPriorityScheduler::with_config(1024, config);
        assert!(!scheduler.config().enable_age_promotion);
    }

    #[test]
    fn test_dynamic_scheduler_add_request_with_deadline() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_request(
            vec![1, 2, 3],
            10,
            Priority::High,
            Some(Deadline::strict(100, 500)),
        );
        assert_eq!(id, 0);
        let request = scheduler.get_request(id).unwrap();
        assert_eq!(request.original_priority, Priority::High);
        assert!(request.deadline.is_some());
    }

    #[test]
    fn test_dynamic_scheduler_queue_depth() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Normal, None);
        scheduler.add_request(vec![3], 5, Priority::High, None);
        scheduler.add_request(vec![4], 5, Priority::Critical, None);

        assert_eq!(scheduler.queue_depth(Priority::Low), 1);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
        assert_eq!(scheduler.queue_depth(Priority::Critical), 1);
    }

    #[test]
    fn test_dynamic_scheduler_stats_accessor() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_dynamic_scheduler_sla_compliance_empty() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        // No requests = 100% compliance
        assert!((scheduler.sla_compliance_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_scheduler_schedule_with_fair_share() {
        let config = DynamicPriorityConfig::default();
        let mut scheduler = DynamicPriorityScheduler::with_config(100, config);

        // Add requests at different priorities
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Normal, None);
        scheduler.add_request(vec![3], 5, Priority::High, None);

        let scheduled = scheduler.schedule(3);
        // Higher priority should be scheduled first
        assert!(!scheduled.is_empty());
    }

    #[test]
    fn test_dynamic_scheduler_schedule_critical_first() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add low priority first
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        // Then critical
        scheduler.add_request(vec![2], 5, Priority::Critical, None);

        let scheduled = scheduler.schedule(1);
        assert_eq!(scheduled.len(), 1);
        // Critical should be scheduled first
        let (id, _tokens) = scheduled[0];
        let request = scheduler.get_request(id).unwrap();
        assert_eq!(request.original_priority, Priority::Critical);
    }

    #[test]
    fn test_dynamic_scheduler_complete_with_sla_met() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with long target latency (will definitely be met)
        let id = scheduler.add_request(
            vec![1],
            5,
            Priority::Normal,
            Some(Deadline::with_target(60000)),
        );
        scheduler.schedule(1);

        scheduler.complete_request(id);
        assert_eq!(scheduler.stats().sla_met, 1);
        assert_eq!(scheduler.stats().sla_missed, 0);
    }

    #[test]
    fn test_dynamic_scheduler_promote_disabled() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        scheduler.add_request(vec![1], 5, Priority::Low, None);

        // Promotion is disabled, so this should be a no-op
        scheduler.promote_aged_requests();

        assert_eq!(scheduler.stats().promotions, 0);
    }

    #[test]
    fn test_dynamic_scheduler_schedule_no_slots() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // Schedule with 0 available slots
        let scheduled = scheduler.schedule(0);
        assert!(scheduled.is_empty());
    }

    #[test]
    fn test_dynamic_scheduler_schedule_no_budget() {
        let mut scheduler =
            DynamicPriorityScheduler::with_config(0, DynamicPriorityConfig::default());
        scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // Zero budget
        let scheduled = scheduler.schedule(10);
        // With 0 budget and min_tokens_per_request=1, nothing should be scheduled
        assert!(scheduled.is_empty());
    }

    // ========================================================================
    // SequenceBatchEntry Tests
    // ========================================================================

    #[test]
    fn test_sequence_batch_entry_new() {
        let entry = SequenceBatchEntry::new(0, 1, 100);
        assert_eq!(entry.seq_idx, 0);
        assert_eq!(entry.slot_id, 1);
        assert_eq!(entry.request_id, 100);
        assert_eq!(entry.position, 0);
        assert!(entry.tokens.is_empty());
        assert!(entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_entry_with_tokens() {
        let entry = SequenceBatchEntry::new(0, 1, 100).with_tokens(vec![1, 2, 3]);
        assert_eq!(entry.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_sequence_batch_entry_at_position() {
        let entry = SequenceBatchEntry::new(0, 1, 100).at_position(42);
        assert_eq!(entry.position, 42);
    }

    #[test]
    fn test_sequence_batch_entry_decoding() {
        let entry = SequenceBatchEntry::new(0, 1, 100).decoding();
        assert!(!entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_entry_builder_chain() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![10, 20])
            .at_position(5)
            .decoding();
        assert_eq!(entry.tokens, vec![10, 20]);
        assert_eq!(entry.position, 5);
        assert!(!entry.is_prefill);
    }

    // ========================================================================
    // SequenceBatch Tests
    // ========================================================================

    #[test]
    fn test_sequence_batch_full() {
        let mut batch = SequenceBatch::new(2);
        assert!(!batch.is_full());

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!(!batch.is_full());

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!(batch.is_full());
    }

    #[test]
    fn test_sequence_batch_add_when_full() {
        let mut batch = SequenceBatch::new(1);
        assert!(batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)));
        assert!(!batch.add_sequence(SequenceBatchEntry::new(1, 1, 2)));
    }

    #[test]
    fn test_sequence_batch_remove_nonexistent() {
        let mut batch = SequenceBatch::new(8);
        batch.add_sequence(SequenceBatchEntry::new(5, 0, 1));
        assert!(batch.remove_sequence(999).is_none());
    }

    #[test]
    fn test_sequence_batch_utilization() {
        let mut batch = SequenceBatch::new(4);
        assert!((batch.utilization - 0.0).abs() < 0.001);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!((batch.utilization - 0.25).abs() < 0.001);

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!((batch.utilization - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sequence_batch_utilization_zero_max() {
        let batch = SequenceBatch::new(0);
        assert!((batch.utilization - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_sequence_batch_num_prefill_decode() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // decode
        batch.add_sequence(SequenceBatchEntry::new(2, 2, 3)); // prefill

        assert_eq!(batch.num_prefill(), 2);
        assert_eq!(batch.num_decode(), 1);
    }

    #[test]
    fn test_sequence_batch_clear() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));

        batch.clear();
        assert!(batch.is_empty());
        assert!((batch.utilization - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // BatchScheduler Tests
    // ========================================================================

    #[test]
    fn test_batch_scheduler_default() {
        let scheduler = BatchScheduler::default();
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_complete_sequence() {
        let mut scheduler = BatchScheduler::new();
        let seq_idx = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();

        assert_eq!(scheduler.num_sequences(), 1);

        let entry = scheduler.complete_sequence(seq_idx);
        assert!(entry.is_some());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_complete_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        let entry = scheduler.complete_sequence(999);
        assert!(entry.is_none());
    }

    #[test]
    fn test_batch_scheduler_start_decode() {
        let mut scheduler = BatchScheduler::new();
        let seq_idx = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();

        assert!(scheduler.start_decode(seq_idx, 3));

        let sbatch = scheduler.sbatch();
        let entry = sbatch.get(seq_idx).unwrap();
        assert!(!entry.is_prefill);
        assert_eq!(entry.position, 3);
        assert!(entry.tokens.is_empty()); // cleared after transitioning
    }

    #[test]
    fn test_batch_scheduler_start_decode_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        assert!(!scheduler.start_decode(999, 0));
    }

    #[test]
    fn test_batch_scheduler_add_when_full() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!(scheduler.add_sequence(0, 1, vec![1]).is_some());
        assert!(scheduler.add_sequence(1, 2, vec![2]).is_some());
        assert!(scheduler.add_sequence(2, 3, vec![3]).is_none()); // Full
    }

    #[test]
    fn test_batch_scheduler_has_capacity() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!(scheduler.has_capacity());
        scheduler.add_sequence(0, 1, vec![1]);
        assert!(scheduler.has_capacity());
        scheduler.add_sequence(1, 2, vec![2]);
        assert!(!scheduler.has_capacity());
    }

    #[test]
    fn test_batch_scheduler_utilization() {
        let config = BatchConfig::default().with_max_sequences(4);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!((scheduler.utilization() - 0.0).abs() < 0.001);

        scheduler.add_sequence(0, 1, vec![1]);
        assert!((scheduler.utilization() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_batch_scheduler_create_empty_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_empty());
    }

    #[test]
    fn test_batch_scheduler_decode_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let seq1 = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();
        let seq2 = scheduler.add_sequence(1, 2, vec![4, 5]).unwrap();

        // Transition both to decode
        scheduler.start_decode(seq1, 3);
        scheduler.start_decode(seq2, 2);

        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_decode());
        assert_eq!(ubatch.n_decode_tokens, 2);
        assert_eq!(ubatch.n_prompt_tokens, 0);
    }

    #[test]
    fn test_batch_scheduler_config_accessor() {
        let config = BatchConfig::default().with_max_tokens(256);
        let scheduler = BatchScheduler::with_config(config);
        assert_eq!(scheduler.config().max_ubatch_tokens, 256);
    }

    // ========================================================================
    // BatchToken Tests
    // ========================================================================

    #[test]
    fn test_batch_token_new() {
        let token = BatchToken::new(42, 1, 5, true);
        assert_eq!(token.token_id, 42);
        assert_eq!(token.seq_idx, 1);
        assert_eq!(token.seq_pos, 5);
        assert!(token.is_prompt);
    }

    #[test]
    fn test_batch_token_clone() {
        let token = BatchToken::new(100, 2, 10, false);
        let cloned = token;
        assert_eq!(cloned.token_id, 100);
        assert_eq!(cloned.seq_idx, 2);
        assert_eq!(cloned.seq_pos, 10);
        assert!(!cloned.is_prompt);
    }

    // ========================================================================
    // MicroBatch Tests
    // ========================================================================

    #[test]
    fn test_micro_batch_token_ids() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));
        batch.add_token(BatchToken::new(30, 0, 2, true));

        assert_eq!(batch.token_ids(), vec![10, 20, 30]);
    }

    #[test]
    fn test_micro_batch_positions() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));
        batch.add_token(BatchToken::new(3, 1, 5, false)); // Different sequence

        assert_eq!(batch.positions(), vec![0, 1, 5]);
    }

    #[test]
    fn test_micro_batch_max_seq_len_tracking() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 5, true));
        assert_eq!(batch.max_seq_len, 6); // pos 5 + 1

        batch.add_token(BatchToken::new(2, 0, 10, true));
        assert_eq!(batch.max_seq_len, 11);

        batch.add_token(BatchToken::new(3, 0, 3, true)); // Lower pos
        assert_eq!(batch.max_seq_len, 11); // Still 11
    }

    // ========================================================================
    // BatchStats Tests
    // ========================================================================

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.ubatches_created, 0);
        assert_eq!(stats.sbatches_created, 0);
        assert_eq!(stats.tokens_processed, 0);
        assert_eq!(stats.prefill_tokens, 0);
        assert_eq!(stats.decode_tokens, 0);
        assert!((stats.avg_ubatch_size - 0.0).abs() < 0.001);
        assert!((stats.avg_sbatch_size - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // SlotManager Additional Tests
    // ========================================================================

    #[test]
    fn test_slot_manager_batch_slots() {
        let mut manager = SlotManager::new(4, 2048);

        // Assign and start generation on some slots
        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);
        manager.assign_request(vec![3], 5);

        // Start generation on slots 0 and 2
        manager.get_slot_mut(0).unwrap().start_generation(1.0);
        manager.get_slot_mut(2).unwrap().start_generation(1.0);

        let batch_slots = manager.batch_slots();
        assert_eq!(batch_slots.len(), 2);
        assert!(batch_slots.contains(&0));
        assert!(batch_slots.contains(&2));
    }

    #[test]
    fn test_slot_manager_active_slots() {
        let mut manager = SlotManager::new(3, 2048);

        // All idle initially
        assert_eq!(manager.active_slots().count(), 0);

        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);

        assert_eq!(manager.active_slots().count(), 2);
    }

    #[test]
    fn test_slot_manager_empty_utilization() {
        let manager = SlotManager::new(0, 2048);
        assert!((manager.utilization() - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Slot Additional Tests
    // ========================================================================

    #[test]
    fn test_slot_tokens_per_second_no_generation_time() {
        let slot = Slot::new(0);
        assert!((slot.tokens_per_second() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_slot_debug() {
        let slot = Slot::new(42);
        let debug_str = format!("{:?}", slot);
        assert!(debug_str.contains("Slot"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_slot_clone() {
        let mut slot = Slot::new(5);
        slot.assign(100, vec![1, 2, 3], 10);

        let cloned = slot.clone();
        assert_eq!(cloned.id, 5);
        assert_eq!(cloned.request_id, Some(100));
        assert_eq!(cloned.input_tokens, vec![1, 2, 3]);
    }

    // ========================================================================
    // Scheduler Preemption Tests
    // ========================================================================

    #[test]
    fn test_scheduler_with_max_tokens() {
        let scheduler = Scheduler::new(32, 1000).with_max_tokens(4096);
        // Just verify it compiles and doesn't panic
        assert!(scheduler.running_count() == 0);
    }

    #[test]
    fn test_scheduler_get_nonexistent_request() {
        let scheduler = Scheduler::new(32, 1000);
        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_scheduler_preempted_count() {
        let scheduler = Scheduler::new(32, 1000);
        assert_eq!(scheduler.preempted_count(), 0);
    }

    #[test]
    fn test_scheduler_update_after_iteration_nonexistent() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut gen = HashMap::new();
        gen.insert(99999u64, 42u32);

        // Should not panic with nonexistent request
        scheduler.update_after_iteration(&gen);
    }

    #[test]
    fn test_scheduler_complete_nonexistent_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Should not panic with nonexistent request
        scheduler.complete_request(99999, &mut kv_cache);
    }
}
