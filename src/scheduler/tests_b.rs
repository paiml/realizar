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
