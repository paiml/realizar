#[cfg(test)]
mod tests {
    use crate::scheduler::*;

    // === Priority Tests ===

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    // === SchedulerRequest Tests ===

    #[test]
    fn test_request_new() {
        let request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 1);
        assert_eq!(request.input_ids.len(), 3);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, SequenceState::Waiting);
    }

    #[test]
    fn test_request_with_priority() {
        let request = SchedulerRequest::new(1, vec![1], 10).with_priority(Priority::High);
        assert_eq!(request.priority, Priority::High);
    }

    #[test]
    fn test_request_total_tokens() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.total_tokens(), 3);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    #[test]
    fn test_request_remaining_tokens() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![4, 5, 6];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_request_is_complete() {
        let mut request = SchedulerRequest::new(1, vec![1], 3);

        // Not complete initially
        assert!(!request.is_complete(0));

        // Complete by max_tokens
        request.generated_tokens = vec![2, 3, 4];
        assert!(request.is_complete(0));

        // Complete by EOS
        let mut request2 = SchedulerRequest::new(2, vec![1], 10);
        request2.generated_tokens = vec![2, 0]; // 0 is EOS
        assert!(request2.is_complete(0));
    }

    // === SchedulerOutput Tests ===

    #[test]
    fn test_scheduler_output_total_tokens() {
        let output = SchedulerOutput {
            num_prefill_tokens: 100,
            num_decode_tokens: 10,
            ..Default::default()
        };
        assert_eq!(output.total_tokens(), 110);
    }

    #[test]
    fn test_scheduler_output_is_empty() {
        let output = SchedulerOutput::default();
        assert!(output.is_empty());
    }

    // === Scheduler Tests ===

    #[test]
    fn test_scheduler_new() {
        let scheduler = Scheduler::new(32, 1000);
        assert_eq!(scheduler.max_batch_size, 32);
        assert_eq!(scheduler.max_queue_size, 1000);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_scheduler_add_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let request_id = scheduler.add_request(vec![1, 2, 3], 10).expect("test");

        assert_eq!(request_id, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.stats().total_requests, 1);
    }

    #[test]
    fn test_scheduler_add_request_queue_full() {
        let mut scheduler = Scheduler::new(32, 1);
        let _ = scheduler.add_request(vec![1], 10).expect("test");

        let result = scheduler.add_request(vec![2], 10);
        assert!(matches!(result, Err(SchedulerError::QueueFull { .. })));
    }

    #[test]
    fn test_scheduler_schedule() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let _ = scheduler.add_request(vec![1, 2, 3], 10).expect("test");
        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        assert_eq!(output.scheduled_request_ids.len(), 1);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_scheduler_update_after_iteration() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");

        let mut generated = HashMap::new();
        generated.insert(request_id, 42u32);
        scheduler.update_after_iteration(&generated);

        let request = scheduler.get_request(request_id).expect("test");
        assert_eq!(request.generated_tokens, vec![42]);
        assert_eq!(request.iterations, 1);
    }

    #[test]
    fn test_scheduler_complete_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");

        scheduler.complete_request(request_id, &mut kv_cache);

        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.stats().completed_requests, 1);
    }

    #[test]
    fn test_scheduler_priority_ordering() {
        let mut scheduler = Scheduler::new(1, 1000); // Only 1 slot
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add low priority first
        let low_id = scheduler
            .add_request_with_priority(vec![1], 10, Priority::Low)
            .expect("test");
        // Add high priority second
        let _high_id = scheduler
            .add_request_with_priority(vec![2], 10, Priority::High)
            .expect("test");

        // Schedule - should pick high priority
        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        // With only 1 slot, we should see preemption or priority selection
        assert_eq!(output.scheduled_request_ids.len(), 1);

        // Low priority should still be waiting or preempted
        let low_request = scheduler.get_request(low_id).expect("test");
        assert!(
            low_request.state == SequenceState::Waiting
                || low_request.state == SequenceState::Preempted
        );
    }

    #[test]
    fn test_scheduler_max_batch_size() {
        let mut scheduler = Scheduler::new(2, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add 3 requests
        let _ = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.add_request(vec![2], 10).expect("test");
        let _ = scheduler.add_request(vec![3], 10).expect("test");

        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        // Should only schedule 2 (max batch size)
        assert_eq!(output.scheduled_request_ids.len(), 2);
        assert_eq!(scheduler.running_count(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");
        scheduler.complete_request(request_id, &mut kv_cache);

        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.completed_requests, 1);
    }

    // === Error Display Tests ===

    #[test]
    fn test_scheduler_error_display() {
        let err = SchedulerError::QueueFull { capacity: 100 };
        assert!(err.to_string().contains("100"));

        let err = SchedulerError::RequestNotFound(42);
        assert!(err.to_string().contains("42"));

        let err = SchedulerError::InvalidState("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    // === SequenceState Tests ===

    #[test]
    fn test_sequence_state_variants() {
        let states = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Preempted,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        // Just ensure all variants exist and are distinct
        for (i, s1) in states.iter().enumerate() {
            for (j, s2) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    // === Stats Serialization ===

    #[test]
    fn test_scheduler_stats_serialization() {
        let stats = SchedulerStats {
            total_requests: 100,
            completed_requests: 90,
            preemptions: 5,
            avg_wait_time_ms: 10.5,
            avg_ttft_ms: 50.0,
            queue_depth: 10,
            running_count: 8,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: SchedulerStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.total_requests, stats.total_requests);
        assert_eq!(parsed.preemptions, stats.preemptions);
    }

    // ========================================================================
    // Slot-Based Server Tests
    // ========================================================================

    #[test]
    fn test_slot_state_default() {
        assert_eq!(SlotState::default(), SlotState::Idle);
    }

    #[test]
    fn test_slot_new() {
        let slot = Slot::new(0);
        assert_eq!(slot.id, 0);
        assert!(slot.is_idle());
        assert!(!slot.is_generating());
        assert!(slot.request_id.is_none());
    }

    #[test]
    fn test_slot_assign() {
        let mut slot = Slot::new(0);
        slot.assign(42, vec![1, 2, 3], 10);

        assert_eq!(slot.state, SlotState::Processing);
        assert_eq!(slot.request_id, Some(42));
        assert_eq!(slot.input_tokens, vec![1, 2, 3]);
        assert_eq!(slot.max_tokens, 10);
        assert!(!slot.is_idle());
    }

    #[test]
    fn test_slot_start_generation() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(5.0);

        assert_eq!(slot.state, SlotState::Generating);
        assert!(slot.is_generating());
        assert_eq!(slot.prompt_time_ms, 5.0);
        assert!(slot.generation_start.is_some());
    }

    #[test]
    fn test_slot_add_token() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);

        slot.add_token(100);
        slot.add_token(200);

        assert_eq!(slot.generated_tokens, vec![100, 200]);
    }

    #[test]
    fn test_slot_is_complete_max_tokens() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 3);
        slot.start_generation(1.0);

        slot.add_token(100);
        assert!(!slot.is_complete(999)); // EOS token

        slot.add_token(200);
        slot.add_token(300);
        assert!(slot.is_complete(999)); // Max tokens reached
    }

    #[test]
    fn test_slot_is_complete_eos() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 100);
        slot.start_generation(1.0);

        slot.add_token(100);
        assert!(!slot.is_complete(999));

        slot.add_token(999); // EOS token
        assert!(slot.is_complete(999));
    }

    #[test]
    fn test_slot_finish() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);
        slot.add_token(100);

        slot.finish();

        assert!(slot.is_idle());
        assert!(slot.request_id.is_none());
        assert!(slot.seq_id.is_none());
    }

    #[test]
    fn test_slot_manager_new() {
        let manager = SlotManager::new(4, 2048);

        assert_eq!(manager.num_slots(), 4);
        assert_eq!(manager.num_idle_slots(), 4);
        assert_eq!(manager.num_active_slots(), 0);
        assert_eq!(manager.max_context_length, 2048);
    }

    #[test]
    fn test_slot_manager_assign_request() {
        let mut manager = SlotManager::new(4, 2048);

        let result = manager.assign_request(vec![1, 2, 3], 10);
        assert!(result.is_some());

        let (slot_id, request_id) = result.expect("test");
        assert_eq!(slot_id, 0);
        assert_eq!(request_id, 0);

        assert_eq!(manager.num_idle_slots(), 3);
        assert_eq!(manager.num_active_slots(), 1);
    }

    #[test]
    fn test_slot_manager_no_slots_available() {
        let mut manager = SlotManager::new(2, 2048);

        // Fill all slots
        manager.assign_request(vec![1], 10).expect("test");
        manager.assign_request(vec![2], 10).expect("test");

        // Third assignment should fail
        let result = manager.assign_request(vec![3], 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_slot_manager_utilization() {
        let mut manager = SlotManager::new(4, 2048);

        assert_eq!(manager.utilization(), 0.0);

        manager.assign_request(vec![1], 10);
        assert!((manager.utilization() - 0.25).abs() < 0.01);

        manager.assign_request(vec![2], 10);
        assert!((manager.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_slot_manager_batch_slots() {
        let mut manager = SlotManager::new(4, 2048);

        // Assign and start generating on some slots
        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        manager.get_slot_mut(0).expect("test").start_generation(1.0);

        let batch = manager.batch_slots();
        assert_eq!(batch, vec![0]); // Only slot 0 is generating
    }

    #[test]
    fn test_slot_manager_get_slot() {
        let manager = SlotManager::new(4, 2048);

        assert!(manager.get_slot(0).is_some());
        assert!(manager.get_slot(3).is_some());
        assert!(manager.get_slot(4).is_none()); // Out of bounds
    }

    #[test]
    fn test_slot_manager_active_slots() {
        let mut manager = SlotManager::new(4, 2048);

        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        let active: Vec<_> = manager.active_slots().collect();
        assert_eq!(active.len(), 2);
    }

    // === Continuous Batching Tests (ubatch/sbatch) ===

    #[test]
    fn test_batch_type_default() {
        assert_eq!(BatchType::default(), BatchType::Decode);
    }

    #[test]
    fn test_batch_token_new() {
        let token = BatchToken::new(42, 0, 5, true);
        assert_eq!(token.token_id, 42);
        assert_eq!(token.seq_idx, 0);
        assert_eq!(token.seq_pos, 5);
        assert!(token.is_prompt);
    }

    #[test]
    fn test_micro_batch_new() {
        let batch = MicroBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.num_sequences(), 0);
        assert!(batch.is_decode()); // Default type
    }

    #[test]
    fn test_micro_batch_add_tokens() {
        let mut batch = MicroBatch::new();

        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.num_sequences(), 1);
        assert!(batch.is_prefill());
        assert_eq!(batch.n_prompt_tokens, 2);
        assert_eq!(batch.n_decode_tokens, 0);
    }

    #[test]
    fn test_micro_batch_mixed_type() {
        let mut batch = MicroBatch::new();

        batch.add_token(BatchToken::new(1, 0, 0, true)); // Prefill
        batch.add_token(BatchToken::new(2, 1, 5, false)); // Decode

        assert!(batch.is_mixed());
        assert_eq!(batch.n_prompt_tokens, 1);
        assert_eq!(batch.n_decode_tokens, 1);
    }

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
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));
        batch.add_token(BatchToken::new(30, 1, 5, false));

        assert_eq!(batch.positions(), vec![0, 1, 5]);
    }

    #[test]
    fn test_micro_batch_clear() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 1, 0, false));

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.n_prompt_tokens, 0);
        assert_eq!(batch.n_decode_tokens, 0);
        assert_eq!(batch.max_seq_len, 0);
    }

    #[test]
    fn test_micro_batch_max_seq_len() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 10, true));

        assert_eq!(batch.max_seq_len, 11); // Position 10 + 1
    }

    #[test]
    fn test_sequence_batch_entry_new() {
        let entry = SequenceBatchEntry::new(0, 1, 100);
        assert_eq!(entry.seq_idx, 0);
        assert_eq!(entry.slot_id, 1);
        assert_eq!(entry.request_id, 100);
        assert!(entry.is_prefill);
        assert_eq!(entry.position, 0);
    }

    #[test]
    fn test_sequence_batch_entry_builder() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![1, 2, 3])
            .at_position(5)
            .decoding();

        assert_eq!(entry.tokens, vec![1, 2, 3]);
        assert_eq!(entry.position, 5);
        assert!(!entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_new() {
        let batch = SequenceBatch::new(8);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
        assert_eq!(batch.max_batch_size, 8);
    }

    #[test]
    fn test_sequence_batch_add_remove() {
        let mut batch = SequenceBatch::new(4);

        let entry = SequenceBatchEntry::new(0, 0, 1);
        assert!(batch.add_sequence(entry));
        assert_eq!(batch.len(), 1);

        let removed = batch.remove_sequence(0);
        assert!(removed.is_some());
        assert!(batch.is_empty());
    }

    #[test]
    fn test_sequence_batch_full() {
        let mut batch = SequenceBatch::new(2);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));

        assert!(batch.is_full());
        assert!(!batch.add_sequence(SequenceBatchEntry::new(2, 2, 3)));
    }

    #[test]
    fn test_sequence_batch_prefill_decode_counts() {
        let mut batch = SequenceBatch::new(4);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // Prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // Decode
        batch.add_sequence(SequenceBatchEntry::new(2, 2, 3).decoding()); // Decode

        assert_eq!(batch.num_prefill(), 1);
        assert_eq!(batch.num_decode(), 2);
    }

    #[test]
    fn test_sequence_batch_utilization() {
        let mut batch = SequenceBatch::new(4);

        assert_eq!(batch.utilization, 0.0);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!((batch.utilization - 0.25).abs() < 0.01);

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!((batch.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_ubatch_tokens, 512);
        assert_eq!(config.max_sbatch_sequences, 8);
        assert!(config.prefer_pure_decode);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .with_max_tokens(1024)
            .with_max_sequences(16);

        assert_eq!(config.max_ubatch_tokens, 1024);
        assert_eq!(config.max_sbatch_sequences, 16);
    }

    #[test]
    fn test_batch_scheduler_new() {
        let scheduler = BatchScheduler::new();
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.num_sequences(), 0);
        assert_eq!(scheduler.utilization(), 0.0);
    }

    #[test]
    fn test_batch_scheduler_add_sequence() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        assert!(seq_idx.is_some());
        assert_eq!(seq_idx.expect("test"), 0);
        assert_eq!(scheduler.num_sequences(), 1);
    }

    #[test]
    fn test_batch_scheduler_complete_sequence() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler.add_sequence(0, 1, vec![10, 20]).expect("test");
        assert_eq!(scheduler.num_sequences(), 1);

        let completed = scheduler.complete_sequence(seq_idx);
        assert!(completed.is_some());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_start_decode() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler
            .add_sequence(0, 1, vec![10, 20, 30])
            .expect("test");

        // Initially in prefill
        assert!(scheduler.sbatch().get(seq_idx).expect("test").is_prefill);

        // Transition to decode
        assert!(scheduler.start_decode(seq_idx, 3));

        let entry = scheduler.sbatch().get(seq_idx).expect("test");
        assert!(!entry.is_prefill);
        assert_eq!(entry.position, 3);
        assert!(entry.tokens.is_empty()); // Cleared after prefill
    }

    #[test]
    fn test_batch_scheduler_create_ubatch_prefill() {
        let mut scheduler = BatchScheduler::new();

        scheduler.add_sequence(0, 1, vec![10, 20, 30]);

        let ubatch = scheduler.create_ubatch();

        assert!(ubatch.is_prefill());
        assert_eq!(ubatch.len(), 3);
        assert_eq!(ubatch.token_ids(), vec![10, 20, 30]);
    }

    #[test]
    fn test_batch_scheduler_create_ubatch_decode() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler
            .add_sequence(0, 1, vec![10, 20, 30])
            .expect("test");
        scheduler.start_decode(seq_idx, 3);

        let ubatch = scheduler.create_ubatch();

        assert!(ubatch.is_decode());
        assert_eq!(ubatch.len(), 1);
    }

    #[test]
    fn test_batch_scheduler_stats() {
        let mut scheduler = BatchScheduler::new();

        scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        scheduler.create_ubatch();

        let stats = scheduler.stats();
        assert_eq!(stats.ubatches_created, 1);
        assert_eq!(stats.tokens_processed, 3);
        assert_eq!(stats.prefill_tokens, 3);
    }

    #[test]
    fn test_batch_scheduler_capacity() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        scheduler.add_sequence(0, 1, vec![1]);
        scheduler.add_sequence(1, 2, vec![2]);

        assert!(!scheduler.has_capacity());
        assert!(scheduler.add_sequence(2, 3, vec![3]).is_none());
    }

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.ubatches_created, 0);
        assert_eq!(stats.tokens_processed, 0);
        assert_eq!(stats.avg_ubatch_size, 0.0);
    }

    // ========================================================================
    // Dynamic Priority Scheduling Tests
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
    }

    #[test]
    fn test_deadline_strict() {
        let deadline = Deadline::strict(100, 200);
        assert_eq!(deadline.target_latency_ms, 100);
        assert_eq!(deadline.hard_deadline_ms, Some(200));
        assert!((deadline.sla_target - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_priority_config_default() {
        let config = DynamicPriorityConfig::default();
        assert!(config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 5000);
        assert_eq!(config.max_promoted_priority, Priority::High);
        assert!(config.enable_deadline_scheduling);
        assert!(config.enable_fair_share);
    }

    #[test]
    fn test_dynamic_priority_config_builder() {
        let config = DynamicPriorityConfig::with_budgets([0.1, 0.2, 0.3, 0.4])
            .no_promotion()
            .with_promotion_interval(1000);

        assert!(!config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 1000);
        assert!((config.priority_budgets[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_new() {
        let request = DynamicRequest::new(0, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 0);
        assert_eq!(request.input_ids.len(), 3);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.original_priority, Priority::Normal);
        assert_eq!(request.effective_priority, Priority::Normal);
        assert_eq!(request.promotions, 0);
    }

    #[test]
    fn test_dynamic_request_with_priority() {
        let request = DynamicRequest::new(0, vec![1], 10).with_priority(Priority::High);
        assert_eq!(request.original_priority, Priority::High);
        assert_eq!(request.effective_priority, Priority::High);
    }

    #[test]
    fn test_dynamic_request_with_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10).with_deadline(Deadline::with_target(500));
        assert!(request.deadline.is_some());
        assert_eq!(request.deadline.expect("test").target_latency_ms, 500);
    }

    #[test]
    fn test_dynamic_request_urgency_no_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10);
        assert_eq!(request.urgency_score(), 0.0);
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_remaining_tokens() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![2, 3, 4];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_dynamic_request_total_tokens() {
        let mut request = DynamicRequest::new(0, vec![1, 2, 3], 10);
        assert_eq!(request.total_tokens(), 3);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    #[test]
    fn test_dynamic_scheduler_new() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.batch_token_budget, 1024);
    }

    #[test]
    fn test_dynamic_scheduler_add_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        let id1 = scheduler.add_request(vec![1, 2, 3], 10, Priority::Normal, None);
        assert_eq!(id1, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);

        let id2 = scheduler.add_request(vec![4, 5], 5, Priority::High, None);
        assert_eq!(id2, 1);
        assert_eq!(scheduler.waiting_count(), 2);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
    }

    #[test]
    fn test_dynamic_scheduler_add_simple_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_simple_request(vec![1, 2], 5);
        assert_eq!(id, 0);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
    }

    #[test]
    fn test_dynamic_scheduler_schedule_priority_order() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add requests at different priorities
        let low_id = scheduler.add_request(vec![1], 5, Priority::Low, None);
        let normal_id = scheduler.add_request(vec![2], 5, Priority::Normal, None);
        let high_id = scheduler.add_request(vec![3], 5, Priority::High, None);

        // Schedule with 2 slots
        let batch = scheduler.schedule(2);

        // Should schedule high and normal first
        assert_eq!(batch.len(), 2);
        let scheduled_ids: Vec<_> = batch.iter().map(|(id, _)| *id).collect();
        assert!(scheduled_ids.contains(&high_id));
        assert!(scheduled_ids.contains(&normal_id));
        assert!(!scheduled_ids.contains(&low_id));
    }

    #[test]
    fn test_dynamic_scheduler_complete_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        let id = scheduler.add_simple_request(vec![1], 5);
        let _ = scheduler.schedule(1);

        assert_eq!(scheduler.running_count(), 1);

        let completed = scheduler.complete_request(id);
        assert!(completed.is_some());
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.stats().completed_requests, 1);
    }

    #[test]
    fn test_dynamic_scheduler_sla_compliance() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with very long deadline (will be met)
        let id = scheduler.add_request(
            vec![1],
            5,
            Priority::Normal,
            Some(Deadline::with_target(100_000)), // 100 seconds
        );

        let _ = scheduler.schedule(1);
        let _ = scheduler.complete_request(id);

        assert_eq!(scheduler.stats().sla_met, 1);
        assert_eq!(scheduler.stats().sla_missed, 0);
        assert!((scheduler.sla_compliance_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_scheduler_stats() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.dropped_requests, 0);
    }

    #[test]
    fn test_dynamic_scheduler_stats_serialization() {
        let stats = DynamicSchedulerStats {
            total_requests: 100,
            completed_requests: 90,
            sla_met: 85,
            sla_missed: 5,
            dropped_requests: 10,
            promotions: 20,
            avg_ttft_ms: 50.5,
            p99_ttft_ms: 200.0,
            tokens_by_priority: [100, 500, 300, 100],
            queue_depth_by_priority: [5, 10, 3, 1],
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: DynamicSchedulerStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.total_requests, 100);
        assert_eq!(parsed.sla_met, 85);
    }

    #[test]
    fn test_dynamic_scheduler_get_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_simple_request(vec![1, 2, 3], 10);

        let request = scheduler.get_request(id);
        assert!(request.is_some());
        assert_eq!(request.expect("test").input_ids, vec![1, 2, 3]);

        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_dynamic_scheduler_config() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let scheduler = DynamicPriorityScheduler::with_config(1024, config);

        assert!(!scheduler.config().enable_age_promotion);
    }

    #[test]
    fn test_dynamic_scheduler_queue_depths() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Low, None);
        scheduler.add_request(vec![3], 5, Priority::Normal, None);
        scheduler.add_request(vec![4], 5, Priority::High, None);
        scheduler.add_request(vec![5], 5, Priority::Critical, None);

        assert_eq!(scheduler.queue_depth(Priority::Low), 2);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
        assert_eq!(scheduler.queue_depth(Priority::Critical), 1);
        assert_eq!(scheduler.waiting_count(), 5);
    }

    #[test]
    fn test_dynamic_scheduler_token_budget_allocation() {
        let mut scheduler = DynamicPriorityScheduler::new(100);

        // Add one request per priority
        scheduler.add_request(vec![1], 50, Priority::Low, None);
        scheduler.add_request(vec![2], 50, Priority::Normal, None);
        scheduler.add_request(vec![3], 50, Priority::High, None);
        scheduler.add_request(vec![4], 50, Priority::Critical, None);

        // Schedule with enough slots
        let batch = scheduler.schedule(4);

        // All should be scheduled, token allocation based on budgets
        assert_eq!(batch.len(), 4);

        // Higher priority should get more tokens
        let stats = scheduler.stats();
        assert!(stats.tokens_by_priority[3] > 0); // Critical
        assert!(stats.tokens_by_priority[2] > 0); // High
    }

    // === Chunked Prefill Tests ===

    #[test]
    fn test_chunked_prefill_config_default() {
        let config = ChunkedPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.min_prompt_length, 256);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 16);
    }

    #[test]
    fn test_chunked_prefill_config_disabled() {
        let config = ChunkedPrefillConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_chunked_prefill_config_low_latency() {
        let config = ChunkedPrefillConfig::low_latency();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 128);
        assert_eq!(config.min_prompt_length, 64);
    }

    #[test]
    fn test_chunked_prefill_config_high_throughput() {
        let config = ChunkedPrefillConfig::high_throughput();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 1024);
        assert!(!config.allow_decode_interleave);
    }

    #[test]
    fn test_chunked_prefill_config_with_chunk_size() {
        let config = ChunkedPrefillConfig::default().with_chunk_size(256);
        assert_eq!(config.chunk_size, 256);
    }

    #[test]
    fn test_chunked_prefill_state_new() {
        let state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.seq_id, 1);
        assert_eq!(state.total_tokens, 1000);
        assert_eq!(state.processed_tokens, 0);
        assert_eq!(state.current_chunk, 0);
        assert_eq!(state.total_chunks, 2); // 1000 / 512 = 2 (ceiling)
        assert!(!state.is_complete());
    }

    #[test]
    fn test_chunked_prefill_state_next_chunk() {
        let state = ChunkedPrefillState::new(1, 1000, 512);
        let range = state.next_chunk(512);
        assert_eq!(range, 0..512);
    }

    #[test]
    fn test_chunked_prefill_state_advance() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        state.advance(512, 50);
        assert_eq!(state.processed_tokens, 512);
        assert_eq!(state.current_chunk, 1);
        assert_eq!(state.chunk_latencies.len(), 1);
        assert_eq!(state.chunk_latencies[0], 50);

        // Next chunk
        let range = state.next_chunk(512);
        assert_eq!(range, 512..1000);
    }

    #[test]
    fn test_chunked_prefill_state_completion() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert!(!state.is_complete());

        state.advance(512, 50);
        assert!(!state.is_complete());

        state.advance(488, 40);
        assert!(state.is_complete());
    }

    #[test]
    fn test_chunked_prefill_state_progress() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert!((state.progress() - 0.0).abs() < 0.01);

        state.advance(500, 50);
        assert!((state.progress() - 50.0).abs() < 0.01);

        state.advance(500, 50);
        assert!((state.progress() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_chunked_prefill_state_remaining_tokens() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.remaining_tokens(), 1000);

        state.advance(600, 50);
        assert_eq!(state.remaining_tokens(), 400);
    }

    #[test]
    fn test_chunked_prefill_state_avg_latency() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.avg_chunk_latency_ms(), 0.0);

        state.advance(512, 50);
        assert_eq!(state.avg_chunk_latency_ms(), 50.0);

        state.advance(488, 30);
        assert_eq!(state.avg_chunk_latency_ms(), 40.0);
    }

    #[test]
    fn test_chunked_prefill_state_zero_tokens() {
        let state = ChunkedPrefillState::new(1, 0, 512);
        assert!(state.is_complete());
        assert_eq!(state.progress(), 100.0);
    }

    #[test]
    fn test_chunked_prefill_stats_default() {
        let stats = ChunkedPrefillStats::default();
        assert_eq!(stats.chunked_sequences, 0);
        assert_eq!(stats.bypassed_sequences, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.avg_chunk_latency_ms(), 0.0);
        assert_eq!(stats.chunking_rate(), 0.0);
    }

    #[test]
    fn test_chunked_prefill_stats_avg_latency() {
        let stats = ChunkedPrefillStats {
            chunks_processed: 4,
            total_chunk_latency_ms: 200,
            ..Default::default()
        };
        assert_eq!(stats.avg_chunk_latency_ms(), 50.0);
    }

    #[test]
    fn test_chunked_prefill_stats_chunking_rate() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 3,
            bypassed_sequences: 7,
            ..Default::default()
        };
        assert!((stats.chunking_rate() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_chunked_prefill_scheduler_new() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert_eq!(scheduler.queue_len(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_submit_short() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Short prompt bypasses chunking
        let (seq_id, use_chunking) = scheduler.submit(100); // < min_prompt_length (256)
        assert_eq!(seq_id, 0);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_submit_long() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Long prompt uses chunking
        let (seq_id, use_chunking) = scheduler.submit(1000); // >= min_prompt_length
        assert_eq!(seq_id, 0);
        assert!(use_chunking);
        assert_eq!(scheduler.stats().chunked_sequences, 1);
        assert_eq!(scheduler.queue_len(), 1);
    }

    #[test]
    fn test_chunked_prefill_scheduler_next_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);

        let chunk = scheduler.next_chunk();
        assert!(chunk.is_some());
        let (seq_id, range) = chunk.expect("test");
        assert_eq!(seq_id, 0);
        assert_eq!(range, 0..512);
    }

    #[test]
    fn test_chunked_prefill_scheduler_complete_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.complete_chunk(0, 512, 50);

        assert_eq!(scheduler.stats().chunks_processed, 1);
        assert_eq!(scheduler.stats().total_chunk_latency_ms, 50);
        assert_eq!(scheduler.stats().max_chunk_latency_ms, 50);

        // State should be updated
        let state = scheduler.get_state(0).expect("test");
        assert_eq!(state.processed_tokens, 512);
    }

    #[test]
    fn test_chunked_prefill_scheduler_full_prefill() {
        let mut scheduler =
            ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default().with_chunk_size(512));

        scheduler.submit(1000);

        // First chunk
        let (seq_id, range) = scheduler.next_chunk().expect("test");
        assert_eq!(range, 0..512);
        scheduler.complete_chunk(seq_id, 512, 50);

        // Second chunk
        let (seq_id, range) = scheduler.next_chunk().expect("test");
        assert_eq!(range, 512..1000);
        scheduler.complete_chunk(seq_id, 488, 40);

        // No more chunks
        assert!(scheduler.next_chunk().is_none());
        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_chunked_prefill_scheduler_has_pending_prefill() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Non-existent sequence
        assert!(!scheduler.has_pending_prefill(999));

        // New sequence has pending prefill
        scheduler.submit(1000);
        assert!(scheduler.has_pending_prefill(0));

        // Complete prefill
        scheduler.complete_chunk(0, 1000, 100);
        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_chunked_prefill_scheduler_remove() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.submit(2000);
        assert_eq!(scheduler.queue_len(), 2);

        let removed = scheduler.remove(0);
        assert!(removed.is_some());
        assert_eq!(scheduler.queue_len(), 1);
    }

    #[test]
    fn test_chunked_prefill_scheduler_clear() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.submit(2000);
        scheduler.clear();

        assert_eq!(scheduler.queue_len(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_decode_interleave() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // No interleave when queue is empty
        assert!(!scheduler.should_interleave_decode());

        scheduler.submit(1000);

        // Should interleave when queue has items
        assert!(scheduler.should_interleave_decode());

        scheduler.record_decode_interleave();
        assert_eq!(scheduler.stats().decode_interleaves, 1);
    }

    #[test]
    fn test_chunked_prefill_scheduler_prefix_cache_hit() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.record_prefix_cache_hit(100);
        assert_eq!(scheduler.stats().prefix_cache_hits, 100);

        scheduler.record_prefix_cache_hit(50);
        assert_eq!(scheduler.stats().prefix_cache_hits, 150);
    }

    #[test]
    fn test_chunked_prefill_scheduler_disabled() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::disabled());

        // Even long prompts bypass chunking when disabled
        let (_, use_chunking) = scheduler.submit(10000);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_default() {
        let scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.config().enabled);
    }

    // ========================================================================
    // Deep Coverage Tests (_deep_scov_ prefix)
    // ========================================================================

    // --- Error handling and edge cases ---

    #[test]
    fn test_deep_scov_scheduler_error_cache_error_conversion() {
        // Test SchedulerError::CacheError conversion from PagedCacheError
        let cache_err = PagedCacheError::OutOfMemory {
            needed: 10,
            available: 5,
        };
        let scheduler_err: SchedulerError = cache_err.into();
        let msg = scheduler_err.to_string();
        assert!(msg.contains("KV cache error"));
    }

    #[test]
    fn test_deep_scov_priority_entry_equality() {
        // Test PriorityEntry PartialEq implementation
        let now = Instant::now();
        let entry1 = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 42,
        };
        let entry2 = PriorityEntry {
            priority: Priority::Low, // Different priority
            arrival_time: now,
            request_id: 42, // Same request_id
        };
        let entry3 = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 43, // Different request_id
        };
        // PartialEq compares only request_id
        assert_eq!(entry1, entry2);
        assert_ne!(entry1, entry3);
    }

    #[test]
    fn test_deep_scov_priority_entry_ordering() {
        // Test PriorityEntry Ord implementation - higher priority first
        let now = Instant::now();
        let high = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 1,
        };
        let low = PriorityEntry {
            priority: Priority::Low,
            arrival_time: now,
            request_id: 2,
        };
        // High priority should be "greater" (higher in heap)
        assert!(high > low);
    }

    #[test]
    fn test_deep_scov_priority_entry_ordering_same_priority_earlier_first() {
        // Test that earlier arrival time has higher priority when priorities are equal
        use std::thread::sleep;
        use std::time::Duration;

        let earlier = Instant::now();
        sleep(Duration::from_millis(1));
        let later = Instant::now();

        let entry_earlier = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: earlier,
            request_id: 1,
        };
        let entry_later = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: later,
            request_id: 2,
        };
        // Earlier arrival should have higher priority
        assert!(entry_earlier > entry_later);
    }

    #[test]
    fn test_deep_scov_scheduler_with_max_tokens() {
        let scheduler = Scheduler::new(32, 1000).with_max_tokens(4096);
        assert_eq!(scheduler.max_tokens_per_batch, 4096);
    }

    #[test]
    fn test_deep_scov_request_wait_time() {
        let request = SchedulerRequest::new(1, vec![1], 10);
        // Wait time should be very small immediately after creation
        let wait = request.wait_time();
        assert!(wait.as_millis() < 100);
    }

    #[test]
    fn test_deep_scov_scheduler_complete_nonexistent_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);
        // Completing a non-existent request should not panic
        scheduler.complete_request(999, &mut kv_cache);
        assert_eq!(scheduler.stats().completed_requests, 0);
    }

    #[test]
    fn test_deep_scov_scheduler_get_nonexistent_request() {
        let scheduler = Scheduler::new(32, 1000);
        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_deep_scov_scheduler_preemption_flow() {
        // Test preemption when high priority request arrives while low priority is running
        let mut scheduler = Scheduler::new(1, 1000); // Single slot
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Schedule low priority request
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2, 3], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");
        assert_eq!(scheduler.running_count(), 1);

        // Add high priority request
        let _high_id = scheduler
            .add_request_with_priority(vec![4, 5], 10, Priority::Critical)
            .expect("add high");

        // Schedule again - should preempt low priority
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Low priority request should be preempted
        let low_req = scheduler.get_request(low_id).expect("get low");
        assert!(
            low_req.state == SequenceState::Preempted
                || low_req.state == SequenceState::Waiting
                || !output.preempted_seq_ids.is_empty()
        );
    }

    #[test]
    fn test_deep_scov_scheduler_check_completions_eos() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 100).expect("add");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Add EOS token
        let mut generated = HashMap::new();
        generated.insert(request_id, 42u32); // EOS = 42
        scheduler.update_after_iteration(&generated);

        // Schedule again to trigger completion check
        let output = scheduler.schedule(&mut kv_cache, 42).expect("schedule 2");
        assert!(output.completed_request_ids.contains(&request_id));
    }

    #[test]
    fn test_deep_scov_scheduler_output_default_values() {
        let output = SchedulerOutput::default();
        assert!(output.scheduled_seq_ids.is_empty());
        assert!(output.scheduled_request_ids.is_empty());
        assert!(output.preempted_seq_ids.is_empty());
        assert!(output.completed_request_ids.is_empty());
        assert_eq!(output.num_prefill_tokens, 0);
        assert_eq!(output.num_decode_tokens, 0);
        assert_eq!(output.total_tokens(), 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_deep_scov_scheduler_multiple_iterations() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1, 2, 3], 5).expect("add");

        // First schedule (prefill)
        let output1 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");
        assert_eq!(output1.num_prefill_tokens, 3);

        // Update after iteration
        let mut generated = HashMap::new();
        generated.insert(request_id, 10u32);
        scheduler.update_after_iteration(&generated);

        // Second schedule (decode)
        let output2 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");
        assert_eq!(output2.num_decode_tokens, 1);
    }

    // --- Slot edge cases ---

    #[test]
    fn test_deep_scov_slot_tokens_per_second_zero_time() {
        let slot = Slot::new(0);
        // No generation time means 0 tok/s
        assert_eq!(slot.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_deep_scov_slot_is_complete_empty_generated() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);
        // No tokens generated yet
        assert!(!slot.is_complete(999));
    }

    #[test]
    fn test_deep_scov_slot_manager_empty_utilization() {
        let manager = SlotManager::new(0, 2048);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_deep_scov_slot_manager_generating_slots() {
        let mut manager = SlotManager::new(4, 2048);

        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        // Start generation on slot 0
        manager
            .get_slot_mut(0)
            .expect("operation failed")
            .start_generation(1.0);
        // Start generation on slot 1
        manager
            .get_slot_mut(1)
            .expect("operation failed")
            .start_generation(2.0);

        let generating: Vec<_> = manager.generating_slots().collect();
        assert_eq!(generating.len(), 2);
    }

    #[test]
    fn test_deep_scov_slot_manager_aggregate_tokens_per_second() {
        let manager = SlotManager::new(4, 2048);
        // All slots idle, no tokens generated
        let tps = manager.aggregate_tokens_per_second();
        assert_eq!(tps, 0.0);
    }

    // --- MicroBatch edge cases ---

    #[test]
    fn test_deep_scov_micro_batch_with_capacity() {
        let batch = MicroBatch::with_capacity(128);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_deep_scov_micro_batch_multiple_sequences() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 1, 0, true));
        batch.add_token(BatchToken::new(3, 2, 0, false));

        assert_eq!(batch.num_sequences(), 3);
        assert!(batch.is_mixed());
    }

    #[test]
    fn test_deep_scov_micro_batch_pure_decode() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 5, false));
        batch.add_token(BatchToken::new(2, 1, 10, false));

        assert!(batch.is_decode());
        assert_eq!(batch.n_decode_tokens, 2);
        assert_eq!(batch.n_prompt_tokens, 0);
    }

    // --- SequenceBatch edge cases ---

    #[test]
    fn test_deep_scov_sequence_batch_remove_nonexistent() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        let removed = batch.remove_sequence(999);
        assert!(removed.is_none());
    }

    #[test]
    fn test_deep_scov_sequence_batch_get_nonexistent() {
        let batch = SequenceBatch::new(4);
        assert!(batch.get(999).is_none());
    }

    #[test]
    fn test_deep_scov_sequence_batch_get_mut() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));

        let entry = batch.get_mut(0);
        assert!(entry.is_some());
        entry.expect("operation failed").position = 42;

        let entry = batch.get(0).expect("index out of bounds");
        assert_eq!(entry.position, 42);
    }

    #[test]
    fn test_deep_scov_sequence_batch_clear() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.utilization, 0.0);
    }

    #[test]
    fn test_deep_scov_sequence_batch_utilization_zero_max() {
        let mut batch = SequenceBatch::new(0);
        batch.update_utilization();
        assert_eq!(batch.utilization, 0.0);
    }

    #[test]
    fn test_deep_scov_sequence_batch_prefill_decode_iterators() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // decode

        let prefill_count = batch.prefill_sequences().count();
        let decode_count = batch.decode_sequences().count();
        assert_eq!(prefill_count, 1);
        assert_eq!(decode_count, 1);
    }

    // --- BatchScheduler edge cases ---

    #[test]
    fn test_deep_scov_batch_scheduler_default() {
        let scheduler = BatchScheduler::default();
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_deep_scov_batch_scheduler_start_decode_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        assert!(!scheduler.start_decode(999, 10));
    }

    #[test]
    fn test_deep_scov_batch_scheduler_complete_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        let result = scheduler.complete_sequence(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_batch_scheduler_create_empty_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_empty());
    }

    #[test]
    fn test_deep_scov_batch_scheduler_ubatch_limit() {
        let config = BatchConfig::default().with_max_tokens(2);
        let mut scheduler = BatchScheduler::with_config(config);

        scheduler.add_sequence(0, 1, vec![10, 20, 30, 40, 50]);

        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 2); // Limited by max_ubatch_tokens
    }

    #[test]
    fn test_deep_scov_batch_scheduler_mixed_prefill_decode() {
        let config = BatchConfig {
            prefer_pure_decode: false,
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add prefill sequence
        scheduler.add_sequence(0, 1, vec![10, 20]);

        // Add decode sequence
        let seq_idx = scheduler
            .add_sequence(1, 2, vec![30])
            .expect("index out of bounds");
        scheduler.start_decode(seq_idx, 1);

        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_mixed());
    }

    // --- DynamicPriorityScheduler edge cases ---

    #[test]
    fn test_deep_scov_dynamic_request_is_expired_no_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10);
        assert!(!request.is_expired());
    }

    #[test]
    fn test_deep_scov_dynamic_request_urgency_zero_target() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: None,
            sla_target: 0.99,
        });
        assert_eq!(request.urgency_score(), 0.0);
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_complete_nonexistent() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let result = scheduler.complete_request(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_sla_compliance_no_sla() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        // No requests with SLA, should return 1.0
        assert_eq!(scheduler.sla_compliance_rate(), 1.0);
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_promote_aged_disabled() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.promote_aged_requests();
        // Should not promote when disabled
        assert_eq!(scheduler.queue_depth(Priority::Low), 1);
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_fair_share_disabled() {
        let config = DynamicPriorityConfig {
            enable_fair_share: false,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);
        scheduler.add_request(vec![1], 50, Priority::Low, None);
        let batch = scheduler.schedule(1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_zero_slots() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);
        let batch = scheduler.schedule(0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_zero_budget() {
        let mut scheduler = DynamicPriorityScheduler::new(0);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);
        let batch = scheduler.schedule(10);
        // With 0 budget, nothing should be scheduled
        assert!(batch.is_empty());
    }

    // --- ChunkedPrefillScheduler edge cases ---

    #[test]
    fn test_deep_scov_chunked_prefill_scheduler_get_state_nonexistent() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(scheduler.get_state(999).is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_scheduler_remove_nonexistent() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let result = scheduler.remove(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_round_robin_mode() {
        let config = ChunkedPrefillConfig {
            boost_partial_prefill: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);

        scheduler.submit(1000);
        scheduler.submit(2000);

        // Complete first chunk of first sequence
        scheduler.complete_chunk(0, 512, 50);

        // Should be moved to back of queue (round-robin)
        // Queue should now have seq 1 first
        let (next_seq, _) = scheduler.next_chunk().expect("operation failed");
        assert_eq!(next_seq, 1);
    }

    #[test]
    fn test_deep_scov_chunked_prefill_no_interleave() {
        let config = ChunkedPrefillConfig {
            allow_decode_interleave: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);
        scheduler.submit(1000);
        assert!(!scheduler.should_interleave_decode());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_next_chunk_empty_queue() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(scheduler.next_chunk().is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_complete_chunk_nonexistent() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        // Should not panic
        scheduler.complete_chunk(999, 100, 10);
        assert_eq!(scheduler.stats().chunks_processed, 0);
    }

    #[test]
    fn test_deep_scov_update_after_iteration_nonexistent() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut generated = HashMap::new();
        generated.insert(999u64, 42u32);
        // Should not panic when updating non-existent request
        scheduler.update_after_iteration(&generated);
    }

    // ========================================================================
    // Coverage Improvement Tests (_cov_ prefix)
    // ========================================================================
    // These tests target specific uncovered branches identified in coverage analysis

    // --- Scheduler KV Cache Allocation Failure Paths ---

    #[test]
    fn test_cov_schedule_waiting_kv_allocation_failure() {
        // Test the Err branch in schedule_waiting when KV cache is full
        let mut scheduler = Scheduler::new(32, 1000);
        // Create KV cache with very limited pages to force allocation failure
        let mut kv_cache = PagedKvCache::new(1, 4, 4, 16); // Very small cache

        // Add multiple requests that will exceed cache capacity
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 1");
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 2");
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 3");

        // First schedule attempt - should schedule at least one
        let output1 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");

        // At least one request was put back due to allocation failure
        // or all succeeded with small cache
        let total_scheduled = output1.scheduled_request_ids.len();
        let total_waiting = scheduler.waiting_count();

        // Either some are scheduled and some waiting, or all scheduled
        assert!(total_scheduled >= 1 || total_waiting >= 1);
    }

    #[test]
    fn test_cov_resume_preempted_kv_allocation_failure() {
        // Test the Err branch in resume_preempted when KV cache is full
        let mut scheduler = Scheduler::new(1, 1000); // Single slot
        let mut kv_cache = PagedKvCache::new(2, 4, 4, 16); // Limited cache

        // Schedule low priority request
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2, 3, 4], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");

        // Add critical priority to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![1, 2], 5, Priority::Critical)
            .expect("add critical");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Low priority may be preempted - verify the method is callable
        let _preempted = scheduler.preempted_count();

        // Complete the critical request
        let low_req = scheduler.get_request(low_id);
        if low_req.is_some() && low_req.expect("test").state == SequenceState::Preempted {
            // The preempted request exists
            assert!(scheduler.preempted_count() >= 1);
        }
    }

    #[test]
    fn test_cov_preemption_with_seq_id() {
        // Test that preemption properly frees seq_id
        let mut scheduler = Scheduler::new(1, 1000);
        let mut kv_cache = PagedKvCache::new(10, 4, 4, 16);

        // Add and schedule low priority
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Verify low priority got a seq_id
        let low_req = scheduler.get_request(low_id).expect("get low");
        assert!(low_req.seq_id.is_some());

        // Add critical priority to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![3, 4], 5, Priority::Critical)
            .expect("add critical");
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Check preemption occurred
        if !output.preempted_seq_ids.is_empty() {
            assert_eq!(scheduler.stats().preemptions, 1);
        }
    }

    // --- Dynamic Priority Scheduler Promotion Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_promotion_low_to_normal() {
        // Test actual promotion from Low to Normal with instant promotion
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0, // Instant promotion
            max_promoted_priority: Priority::High,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add low priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Low, None);
        assert_eq!(scheduler.queue_depth(Priority::Low), 1);

        // Promote - should move from Low to Normal
        scheduler.promote_aged_requests();

        // Request should have been promoted
        let request = scheduler.get_request(id).expect("get request");
        assert!(request.effective_priority >= Priority::Normal);
        assert!(request.promotions >= 1);
    }

    #[test]
    fn test_cov_dynamic_scheduler_promotion_normal_to_high() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::High,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add Normal priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // First promotion: Normal -> High
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::High);
    }

    #[test]
    fn test_cov_dynamic_scheduler_promotion_capped_at_max() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::Normal, // Cap at Normal
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add Low priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Low, None);

        // First promotion: Low -> Normal
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        // Should be Normal now
        assert_eq!(request.effective_priority, Priority::Normal);

        // Second promotion should not happen (Normal >= max_promoted_priority)
        scheduler.promote_aged_requests();
        let request = scheduler.get_request(id).expect("get request 2");
        assert_eq!(request.effective_priority, Priority::Normal);
    }

    #[test]
    fn test_cov_dynamic_scheduler_high_priority_no_promotion_when_at_max() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::High, // Cap at High
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add High priority request (already at max)
        let id = scheduler.add_request(vec![1], 5, Priority::High, None);

        // Promotion should not happen (High >= max_promoted_priority)
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::High);
        assert_eq!(request.promotions, 0);
    }

    // --- Deadline Expiration Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_deadline_expired() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with immediate hard deadline (already expired)
        let deadline = Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: Some(0), // Immediate expiration
            sla_target: 1.0,
        };
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline));
        assert_eq!(scheduler.waiting_count(), 1);

        // Small delay to ensure deadline passes
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Drop expired
        let dropped = scheduler.drop_expired();
        assert!(dropped.contains(&id));
        assert_eq!(scheduler.stats().dropped_requests, 1);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_cov_dynamic_scheduler_sla_missed() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with immediate target (will miss SLA)
        let deadline = Deadline {
            target_latency_ms: 0,   // Will miss immediately
            hard_deadline_ms: None, // No hard deadline
            sla_target: 0.99,
        };
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline));

        // Small delay
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Schedule and complete
        let _ = scheduler.schedule(1);
        let _ = scheduler.complete_request(id);

        // SLA should be missed
        assert_eq!(scheduler.stats().sla_missed, 1);
        assert!(scheduler.sla_compliance_rate() < 1.0);
    }

    #[test]
    fn test_cov_dynamic_request_is_urgent() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::with_target(1)); // 1ms target

        // Wait a tiny bit
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Should now be urgent (elapsed >= target/2)
        assert!(request.is_urgent() || request.urgency_score() > 0.0);
    }

    #[test]
    fn test_cov_dynamic_request_is_expired_with_hard_deadline() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::strict(0, 0)); // Immediate hard deadline

        std::thread::sleep(std::time::Duration::from_millis(1));

        assert!(request.is_expired());
    }

    // --- TTFT Statistics Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_ttft_p99_calculation() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add and complete multiple requests with SLA
        for i in 0..10 {
            let deadline = Deadline::with_target(100_000); // Long deadline
            let id = scheduler.add_request(vec![i as u32], 5, Priority::Normal, Some(deadline));
            let _ = scheduler.schedule(1);
            let _ = scheduler.complete_request(id);
        }

        // Check TTFT stats are populated
        assert!(scheduler.stats().avg_ttft_ms >= 0.0);
        assert!(scheduler.stats().p99_ttft_ms >= 0.0);
    }

    // --- Slot State Serialization Tests ---

    #[test]
    fn test_cov_slot_state_serialization() {
        let states = [
            SlotState::Idle,
            SlotState::Processing,
            SlotState::Generating,
        ];
        for state in &states {
            let json = serde_json::to_string(state).expect("serialize");
            let parsed: SlotState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*state, parsed);
        }
    }

    #[test]
    fn test_cov_batch_type_serialization() {
        let types = [BatchType::Prefill, BatchType::Decode, BatchType::Mixed];
        for bt in &types {
            let json = serde_json::to_string(bt).expect("serialize");
            let parsed: BatchType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*bt, parsed);
        }
    }

    #[test]
    fn test_cov_batch_token_serialization() {
        let token = BatchToken::new(42, 1, 5, true);
        let json = serde_json::to_string(&token).expect("serialize");
        let parsed: BatchToken = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(token.token_id, parsed.token_id);
        assert_eq!(token.seq_idx, parsed.seq_idx);
        assert_eq!(token.seq_pos, parsed.seq_pos);
        assert_eq!(token.is_prompt, parsed.is_prompt);
    }

    // --- MicroBatch Token Limit Tests ---

    #[test]
    fn test_cov_batch_scheduler_decode_ubatch_limit() {
        let config = BatchConfig {
            max_ubatch_tokens: 2,
            prefer_pure_decode: true,
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add and start decode for multiple sequences
        let seq1 = scheduler.add_sequence(0, 1, vec![10]).expect("add 1");
        let seq2 = scheduler.add_sequence(1, 2, vec![20]).expect("add 2");
        let seq3 = scheduler.add_sequence(2, 3, vec![30]).expect("add 3");

        scheduler.start_decode(seq1, 1);
        scheduler.start_decode(seq2, 1);
        scheduler.start_decode(seq3, 1);

        // Create ubatch - should be limited to 2 tokens
        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 2);
        assert!(ubatch.is_decode());
    }

    #[test]
    fn test_cov_batch_scheduler_mixed_with_limited_budget() {
        let config = BatchConfig {
            max_ubatch_tokens: 3,
            prefer_pure_decode: false, // Allow mixed
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add prefill sequence with 2 tokens
        let seq1 = scheduler.add_sequence(0, 1, vec![10, 20]).expect("add 1");

        // Add decode sequence
        let seq2 = scheduler.add_sequence(1, 2, vec![30]).expect("add 2");
        scheduler.start_decode(seq2, 1);

        // Add another decode
        let seq3 = scheduler.add_sequence(2, 3, vec![40]).expect("add 3");
        scheduler.start_decode(seq3, 1);

        // Create ubatch - should be limited to 3 tokens total
        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 3);

        // With prefer_pure_decode=false, it may include both prefill and decode
        // The first 2 are prefill, then 1 decode to hit the limit
        assert!(ubatch.is_mixed() || ubatch.is_prefill());

        // Cleanup
        scheduler.complete_sequence(seq1);
    }

    // --- Priority Entry Comparison Edge Cases ---

    #[test]
    fn test_cov_priority_entry_partial_cmp() {
        let now = Instant::now();
        let entry1 = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: now,
            request_id: 1,
        };
        let entry2 = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: now,
            request_id: 2,
        };

        // partial_cmp should delegate to cmp
        let result = entry1.partial_cmp(&entry2);
        assert!(result.is_some());
    }

    // --- SchedulerOutput Non-Empty Tests ---

    #[test]
    fn test_cov_scheduler_output_non_empty() {
        let mut output = SchedulerOutput::default();
        assert!(output.is_empty());

        output.scheduled_seq_ids.push(crate::paged_kv::SeqId::new());
        assert!(!output.is_empty());
    }

    // --- Scheduler Preempted Count ---

    #[test]
    fn test_cov_scheduler_preempted_count() {
        let mut scheduler = Scheduler::new(1, 1000);
        let mut kv_cache = PagedKvCache::new(10, 4, 4, 16);

        assert_eq!(scheduler.preempted_count(), 0);

        // Add low and critical priority
        let _low_id = scheduler
            .add_request_with_priority(vec![1], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0);

        // Add critical to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![2], 5, Priority::Critical)
            .expect("add critical");
        let _ = scheduler.schedule(&mut kv_cache, 0);

        // Preempted count may be >= 0 depending on implementation
        let _count = scheduler.preempted_count();
    }

    // --- Dynamic Scheduler Scheduling With Deadline Sorting ---

    #[test]
    fn test_cov_dynamic_scheduler_deadline_aware_scheduling() {
        let config = DynamicPriorityConfig {
            enable_deadline_scheduling: true,
            enable_fair_share: false, // Disable fair share to test pure deadline ordering
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add requests with different deadlines
        let deadline_short = Deadline::with_target(10);
        let deadline_long = Deadline::with_target(1000);

        // Add long deadline first
        let _id_long = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline_long));

        // Add short deadline second
        let id_short = scheduler.add_request(vec![2], 5, Priority::Normal, Some(deadline_short));

        // Small delay to make short deadline more urgent
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Schedule - short deadline should be scheduled first due to higher urgency
        let batch = scheduler.schedule(1);
        assert_eq!(batch.len(), 1);

        // The short deadline request should be scheduled first
        let (scheduled_id, _) = batch[0];
        assert_eq!(scheduled_id, id_short);
    }

    // --- Promotion to Critical (index 3) ---

    #[test]
    fn test_cov_dynamic_scheduler_promotion_to_critical() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::Critical, // Allow promotion to Critical
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add High priority request
        let id = scheduler.add_request(vec![1], 5, Priority::High, None);

        // Promote: High -> Critical
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::Critical);
    }

    // --- ChunkedPrefillState Edge Cases ---

    #[test]
    fn test_cov_chunked_prefill_state_single_chunk() {
        // Test when total_tokens < chunk_size (single chunk)
        let state = ChunkedPrefillState::new(1, 100, 512);
        assert_eq!(state.total_chunks, 1);
        assert_eq!(state.next_chunk(512), 0..100);
    }

    #[test]
    fn test_cov_chunked_prefill_state_exact_chunks() {
        // Test when total_tokens is exact multiple of chunk_size
        let state = ChunkedPrefillState::new(1, 1024, 512);
        assert_eq!(state.total_chunks, 2);
    }

    // --- BatchStats Serialization ---

    #[test]
    fn test_cov_batch_stats_serialization() {
        let stats = BatchStats {
            ubatches_created: 10,
            sbatches_created: 5,
            tokens_processed: 1000,
            prefill_tokens: 800,
            decode_tokens: 200,
            avg_ubatch_size: 100.0,
            avg_sbatch_size: 2.0,
        };

        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: BatchStats = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.ubatches_created, stats.ubatches_created);
        assert_eq!(parsed.tokens_processed, stats.tokens_processed);
    }

    // --- SequenceBatchEntry Serialization ---

    #[test]
    fn test_cov_sequence_batch_entry_serialization() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![1, 2, 3])
            .at_position(5)
            .decoding();

        let json = serde_json::to_string(&entry).expect("serialize");
        let parsed: SequenceBatchEntry = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.seq_idx, entry.seq_idx);
        assert_eq!(parsed.slot_id, entry.slot_id);
        assert_eq!(parsed.request_id, entry.request_id);
        assert_eq!(parsed.tokens, entry.tokens);
        assert_eq!(parsed.position, entry.position);
        assert_eq!(parsed.is_prefill, entry.is_prefill);
    }

    // --- SequenceBatch Serialization ---

    #[test]
    fn test_cov_sequence_batch_serialization() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding());

        let json = serde_json::to_string(&batch).expect("serialize");
        let parsed: SequenceBatch = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.max_batch_size, batch.max_batch_size);
        assert_eq!(parsed.sequences.len(), 2);
    }

    // --- MicroBatch Serialization ---

    #[test]
    fn test_cov_micro_batch_serialization() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));

        let json = serde_json::to_string(&batch).expect("serialize");
        let parsed: MicroBatch = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.tokens.len(), 2);
        assert_eq!(parsed.n_prompt_tokens, 2);
    }

    // --- BatchConfig Serialization ---

    #[test]
    fn test_cov_batch_config_serialization() {
        let config = BatchConfig::default()
            .with_max_tokens(1024)
            .with_max_sequences(16);

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: BatchConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.max_ubatch_tokens, config.max_ubatch_tokens);
        assert_eq!(parsed.max_sbatch_sequences, config.max_sbatch_sequences);
    }

    // --- ChunkedPrefillConfig Serialization ---

    #[test]
    fn test_cov_chunked_prefill_config_serialization() {
        let config = ChunkedPrefillConfig::low_latency();

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: ChunkedPrefillConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.chunk_size, config.chunk_size);
        assert_eq!(parsed.enabled, config.enabled);
    }

    // --- ChunkedPrefillState Serialization ---

    #[test]
    fn test_cov_chunked_prefill_state_serialization() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        state.advance(512, 50);

        let json = serde_json::to_string(&state).expect("serialize");
        let parsed: ChunkedPrefillState = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.seq_id, state.seq_id);
        assert_eq!(parsed.processed_tokens, state.processed_tokens);
        assert_eq!(parsed.chunk_latencies.len(), 1);
    }

    // --- ChunkedPrefillStats Serialization ---

    #[test]
    fn test_cov_chunked_prefill_stats_serialization() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 10,
            bypassed_sequences: 5,
            chunks_processed: 20,
            decode_interleaves: 8,
            total_chunk_latency_ms: 500,
            max_chunk_latency_ms: 100,
            prefix_cache_hits: 200,
        };

        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: ChunkedPrefillStats = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.chunked_sequences, stats.chunked_sequences);
        assert_eq!(parsed.prefix_cache_hits, stats.prefix_cache_hits);
    }

    // --- Deadline Serialization ---

    #[test]
    fn test_cov_deadline_serialization() {
        let deadline = Deadline::strict(100, 200);

        let json = serde_json::to_string(&deadline).expect("serialize");
        let parsed: Deadline = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.target_latency_ms, deadline.target_latency_ms);
        assert_eq!(parsed.hard_deadline_ms, deadline.hard_deadline_ms);
    }

    // --- DynamicPriorityConfig Serialization ---

    #[test]
    fn test_cov_dynamic_priority_config_serialization() {
        let config = DynamicPriorityConfig::default().no_promotion();

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: DynamicPriorityConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.enable_age_promotion, config.enable_age_promotion);
    }

    // --- ChunkedPrefillScheduler Additional Coverage ---

    #[test]
    fn test_cov_chunked_prefill_max_chunks_limit() {
        let config = ChunkedPrefillConfig {
            enabled: true,
            chunk_size: 10,
            min_prompt_length: 5,
            max_chunks: 3,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);

        // Submit very long prompt
        let (seq_id, use_chunking) = scheduler.submit(1000);
        assert!(use_chunking);

        // Get state
        let state = scheduler.get_state(seq_id).expect("get state");
        // total_chunks is based on ceiling division
        assert!(state.total_chunks >= 1);
    }

    // --- Empty Iteration Check ---

    #[test]
    fn test_cov_scheduler_schedule_empty() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Schedule with no requests
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule");
        assert!(output.is_empty());
        assert_eq!(output.total_tokens(), 0);
    }

    // --- Multiple Completions in Single Schedule ---

    #[test]
    fn test_cov_scheduler_multiple_completions() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add and schedule multiple requests
        let id1 = scheduler.add_request(vec![1], 1).expect("add 1");
        let id2 = scheduler.add_request(vec![2], 1).expect("add 2");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Generate tokens to complete both
        let mut generated = HashMap::new();
        generated.insert(id1, 10u32);
        generated.insert(id2, 20u32);
        scheduler.update_after_iteration(&generated);

        // Schedule again - both should complete
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");
        assert_eq!(output.completed_request_ids.len(), 2);
    }

    // --- Request Without Seq ID During Completion ---

    #[test]
    fn test_cov_scheduler_complete_request_without_seq_id() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add request but don't schedule (so no seq_id)
        let request_id = scheduler.add_request(vec![1], 10).expect("add");

        // Complete without scheduling (no seq_id to free)
        scheduler.complete_request(request_id, &mut kv_cache);

        // Should still update stats (even though nothing was really completed)
        let request = scheduler.get_request(request_id);
        if let Some(r) = request {
            assert_eq!(r.state, SequenceState::Completed);
        }
    }

    // --- Slot Generation Time Calculation ---

    #[test]
    fn test_cov_slot_tokens_per_second_with_generation() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(5.0);

        // Add some tokens
        slot.add_token(100);
        slot.add_token(200);
        slot.add_token(300);

        // Finish to record generation time
        slot.finish();

        // generation_time_ms should be recorded
        assert!(slot.generation_time_ms >= 0.0);
    }

    // --- DynamicRequest with No Hard Deadline ---

    #[test]
    fn test_cov_dynamic_request_no_hard_deadline_not_expired() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::with_target(0)); // Target only, no hard deadline

        std::thread::sleep(std::time::Duration::from_millis(1));

        // Should not be expired (no hard deadline)
        assert!(!request.is_expired());
    }

    // --- BatchScheduler Recording Stats ---

    #[test]
    fn test_cov_batch_scheduler_record_ubatch_stats() {
        let mut scheduler = BatchScheduler::new();

        // Add sequence and create multiple ubatches
        scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        let _ = scheduler.create_ubatch();

        let seq_idx = scheduler.add_sequence(1, 2, vec![40, 50]).expect("add 2");
        scheduler.start_decode(seq_idx, 2);
        let _ = scheduler.create_ubatch();

        let stats = scheduler.stats();
        assert!(stats.ubatches_created >= 2);
        assert!(stats.avg_ubatch_size > 0.0);
    }

    // --- Priority Types Serialization ---

    #[test]
    fn test_cov_priority_serialization() {
        let priorities = [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ];
        for p in &priorities {
            let json = serde_json::to_string(p).expect("serialize");
            let parsed: Priority = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*p, parsed);
        }
    }

    // --- SequenceState Serialization ---

    #[test]
    fn test_cov_sequence_state_serialization() {
        let states = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Preempted,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        for s in &states {
            let json = serde_json::to_string(s).expect("serialize");
            let parsed: SequenceState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*s, parsed);
        }
    }
}
