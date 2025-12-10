//! Continuous Batching Scheduler
//!
//! Per spec §8: Implements continuous batching for LLM serving based on vLLM.
//! Reference: [8] Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based Generative Models"
//!
//! ## Key Features
//!
//! - **Iteration-Level Scheduling**: New requests join batch at any iteration
//! - **Preemption**: Low-priority requests can be preempted for high-priority
//! - **Memory-Aware**: Respects KV cache limits when scheduling
//! - **Fair Queuing**: Prevents starvation of long requests
//!
//! ## Scheduling Algorithm
//!
//! ```text
//! while running:
//!   1. Check for completed sequences (EOS or max_tokens)
//!   2. Preempt sequences if memory pressure
//!   3. Schedule waiting sequences if space available
//!   4. Run one iteration of generation for batch
//! ```

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_wraps)] // Result wrapping for API consistency
#![allow(clippy::derivable_impls)] // Manual impl for documentation clarity
#![allow(clippy::option_if_let_else)] // map_or is more readable

use crate::paged_kv::{PagedCacheError, PagedKvCache, SeqId};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::Instant;
use thiserror::Error;

/// Error type for scheduler operations
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// Queue is full
    #[error("Request queue full: capacity {capacity}")]
    QueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Request not found
    #[error("Request not found: {0}")]
    RequestNotFound(u64),

    /// KV cache error
    #[error("KV cache error: {0}")]
    CacheError(#[from] PagedCacheError),

    /// Invalid state
    #[error("Invalid scheduler state: {0}")]
    InvalidState(String),
}

/// Request priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority (background tasks)
    Low = 0,
    /// Normal priority (standard requests)
    Normal = 1,
    /// High priority (interactive requests)
    High = 2,
    /// Critical priority (system requests)
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Sequence state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceState {
    /// Waiting in queue
    Waiting,
    /// Currently running
    Running,
    /// Preempted (swapped out)
    Preempted,
    /// Completed (EOS or max_tokens)
    Completed,
    /// Failed (error during generation)
    Failed,
}

/// Generation request
#[derive(Debug, Clone)]
pub struct SchedulerRequest {
    /// Unique request ID
    pub request_id: u64,
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Priority level
    pub priority: Priority,
    /// Arrival time
    pub arrival_time: Instant,
    /// Sequence ID (assigned when scheduled)
    pub seq_id: Option<SeqId>,
    /// Current state
    pub state: SequenceState,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Number of decode iterations
    pub iterations: usize,
}

impl SchedulerRequest {
    /// Create a new request
    pub fn new(request_id: u64, input_ids: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            request_id,
            input_ids,
            max_tokens,
            priority: Priority::default(),
            arrival_time: Instant::now(),
            seq_id: None,
            state: SequenceState::Waiting,
            generated_tokens: Vec::new(),
            iterations: 0,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Total tokens (input + generated)
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len() + self.generated_tokens.len()
    }

    /// Remaining tokens to generate
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.generated_tokens.len())
    }

    /// Check if generation is complete
    pub fn is_complete(&self, eos_token: u32) -> bool {
        self.generated_tokens.len() >= self.max_tokens
            || self.generated_tokens.last() == Some(&eos_token)
    }

    /// Time waiting in queue
    pub fn wait_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// Priority-aware entry for the waiting queue
#[derive(Debug)]
struct PriorityEntry {
    priority: Priority,
    arrival_time: Instant,
    request_id: u64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier arrival
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.arrival_time.cmp(&self.arrival_time),
            other => other,
        }
    }
}

/// Scheduler output for one iteration
#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    /// Sequences to run this iteration
    pub scheduled_seq_ids: Vec<SeqId>,
    /// Request IDs for scheduled sequences
    pub scheduled_request_ids: Vec<u64>,
    /// Sequences that were preempted
    pub preempted_seq_ids: Vec<SeqId>,
    /// Sequences that completed
    pub completed_request_ids: Vec<u64>,
    /// Number of tokens in prefill phase
    pub num_prefill_tokens: usize,
    /// Number of tokens in decode phase
    pub num_decode_tokens: usize,
}

impl SchedulerOutput {
    /// Total tokens scheduled
    pub fn total_tokens(&self) -> usize {
        self.num_prefill_tokens + self.num_decode_tokens
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.scheduled_seq_ids.is_empty()
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Total requests received
    pub total_requests: u64,
    /// Total requests completed
    pub completed_requests: u64,
    /// Total preemptions
    pub preemptions: u64,
    /// Average queue wait time (ms)
    pub avg_wait_time_ms: f64,
    /// Average time-to-first-token (ms)
    pub avg_ttft_ms: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Current running count
    pub running_count: usize,
}

/// Continuous batching scheduler
pub struct Scheduler {
    /// All requests by ID
    requests: HashMap<u64, SchedulerRequest>,
    /// Waiting queue (priority-ordered)
    waiting_queue: BinaryHeap<PriorityEntry>,
    /// Running sequences
    running: Vec<u64>,
    /// Preempted sequences (can be resumed)
    preempted: VecDeque<u64>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum queue size
    max_queue_size: usize,
    /// Maximum tokens per batch
    max_tokens_per_batch: usize,
    /// Next request ID
    next_request_id: u64,
    /// Statistics
    stats: SchedulerStats,
    /// Total wait time for completed requests (for averaging)
    total_wait_time_ms: f64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(max_batch_size: usize, max_queue_size: usize) -> Self {
        Self {
            requests: HashMap::new(),
            waiting_queue: BinaryHeap::new(),
            running: Vec::new(),
            preempted: VecDeque::new(),
            max_batch_size,
            max_queue_size,
            max_tokens_per_batch: max_batch_size * 2048, // Default: assume 2k context
            next_request_id: 0,
            stats: SchedulerStats::default(),
            total_wait_time_ms: 0.0,
        }
    }

    /// Set maximum tokens per batch
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_batch = max_tokens;
        self
    }

    /// Add a new request to the queue
    pub fn add_request(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
    ) -> Result<u64, SchedulerError> {
        self.add_request_with_priority(input_ids, max_tokens, Priority::Normal)
    }

    /// Add a request with priority
    pub fn add_request_with_priority(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        priority: Priority,
    ) -> Result<u64, SchedulerError> {
        if self.waiting_queue.len() >= self.max_queue_size {
            return Err(SchedulerError::QueueFull {
                capacity: self.max_queue_size,
            });
        }

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request =
            SchedulerRequest::new(request_id, input_ids, max_tokens).with_priority(priority);
        let entry = PriorityEntry {
            priority,
            arrival_time: request.arrival_time,
            request_id,
        };

        self.requests.insert(request_id, request);
        self.waiting_queue.push(entry);
        self.stats.total_requests += 1;
        self.stats.queue_depth = self.waiting_queue.len();

        Ok(request_id)
    }

    /// Schedule one iteration of generation
    pub fn schedule(
        &mut self,
        kv_cache: &mut PagedKvCache,
        eos_token: u32,
    ) -> Result<SchedulerOutput, SchedulerError> {
        let mut output = SchedulerOutput::default();

        // 1. Check for completed sequences
        self.check_completions(&mut output, eos_token);

        // 2. Preempt if memory pressure (simplified: check if we can fit new sequences)
        self.handle_preemption(&mut output, kv_cache);

        // 3. Resume preempted sequences if possible
        self.resume_preempted(&mut output, kv_cache)?;

        // 4. Schedule new sequences from waiting queue
        self.schedule_waiting(&mut output, kv_cache)?;

        // 5. Build final output
        for &request_id in &self.running {
            if let Some(request) = self.requests.get(&request_id) {
                if let Some(seq_id) = request.seq_id {
                    output.scheduled_seq_ids.push(seq_id);
                    output.scheduled_request_ids.push(request_id);

                    if request.iterations == 0 {
                        // Prefill phase
                        output.num_prefill_tokens += request.input_ids.len();
                    } else {
                        // Decode phase (1 token per sequence)
                        output.num_decode_tokens += 1;
                    }
                }
            }
        }

        self.stats.running_count = self.running.len();
        self.stats.queue_depth = self.waiting_queue.len();

        Ok(output)
    }

    /// Update scheduler after generation iteration
    pub fn update_after_iteration(&mut self, generated_tokens: &HashMap<u64, u32>) {
        for (&request_id, &token) in generated_tokens {
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.generated_tokens.push(token);
                request.iterations += 1;
            }
        }
    }

    /// Mark request as complete
    pub fn complete_request(&mut self, request_id: u64, kv_cache: &mut PagedKvCache) {
        if let Some(request) = self.requests.get_mut(&request_id) {
            request.state = SequenceState::Completed;

            // Free KV cache
            if let Some(seq_id) = request.seq_id {
                kv_cache.free_sequence(seq_id);
            }

            // Update stats
            self.stats.completed_requests += 1;
            let wait_time = request.wait_time().as_secs_f64() * 1000.0;
            self.total_wait_time_ms += wait_time;
            self.stats.avg_wait_time_ms =
                self.total_wait_time_ms / self.stats.completed_requests as f64;
        }

        // Remove from running
        self.running.retain(|&id| id != request_id);
    }

    /// Get request by ID
    pub fn get_request(&self, request_id: u64) -> Option<&SchedulerRequest> {
        self.requests.get(&request_id)
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Check for completed sequences
    fn check_completions(&mut self, output: &mut SchedulerOutput, eos_token: u32) {
        let completed: Vec<u64> = self
            .running
            .iter()
            .filter(|&&id| {
                self.requests
                    .get(&id)
                    .is_some_and(|r| r.is_complete(eos_token))
            })
            .copied()
            .collect();

        for request_id in completed {
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.state = SequenceState::Completed;
            }
            output.completed_request_ids.push(request_id);
        }
    }

    /// Handle preemption under memory pressure
    fn handle_preemption(&mut self, output: &mut SchedulerOutput, kv_cache: &mut PagedKvCache) {
        // Simple preemption: if running at max and waiting queue has higher priority
        if self.running.len() >= self.max_batch_size && !self.waiting_queue.is_empty() {
            // Check if waiting has higher priority than lowest running
            if let Some(waiting_entry) = self.waiting_queue.peek() {
                let min_running_priority = self
                    .running
                    .iter()
                    .filter_map(|&id| self.requests.get(&id))
                    .map(|r| r.priority)
                    .min()
                    .unwrap_or(Priority::Critical);

                if waiting_entry.priority > min_running_priority {
                    // Find lowest priority running request to preempt
                    if let Some(&preempt_id) = self.running.iter().find(|&&id| {
                        self.requests
                            .get(&id)
                            .is_some_and(|r| r.priority == min_running_priority)
                    }) {
                        // Preempt the request
                        if let Some(request) = self.requests.get_mut(&preempt_id) {
                            request.state = SequenceState::Preempted;
                            if let Some(seq_id) = request.seq_id {
                                output.preempted_seq_ids.push(seq_id);
                                kv_cache.free_sequence(seq_id);
                            }
                            request.seq_id = None;
                        }
                        self.running.retain(|&id| id != preempt_id);
                        self.preempted.push_back(preempt_id);
                        self.stats.preemptions += 1;
                    }
                }
            }
        }
    }

    /// Resume preempted sequences
    fn resume_preempted(
        &mut self,
        _output: &mut SchedulerOutput,
        kv_cache: &mut PagedKvCache,
    ) -> Result<(), SchedulerError> {
        while self.running.len() < self.max_batch_size {
            if let Some(request_id) = self.preempted.pop_front() {
                if let Some(request) = self.requests.get_mut(&request_id) {
                    // Try to allocate KV cache
                    let total_tokens = request.total_tokens();
                    match kv_cache.allocate_sequence(total_tokens) {
                        Ok(seq_id) => {
                            request.seq_id = Some(seq_id);
                            request.state = SequenceState::Running;
                            self.running.push(request_id);
                        },
                        Err(_) => {
                            // Can't allocate, put back in preempted queue
                            self.preempted.push_front(request_id);
                            break;
                        },
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Schedule waiting requests
    fn schedule_waiting(
        &mut self,
        _output: &mut SchedulerOutput,
        kv_cache: &mut PagedKvCache,
    ) -> Result<(), SchedulerError> {
        while self.running.len() < self.max_batch_size {
            if let Some(entry) = self.waiting_queue.pop() {
                if let Some(request) = self.requests.get_mut(&entry.request_id) {
                    let total_tokens = request.input_ids.len();
                    match kv_cache.allocate_sequence(total_tokens) {
                        Ok(seq_id) => {
                            request.seq_id = Some(seq_id);
                            request.state = SequenceState::Running;
                            self.running.push(entry.request_id);
                        },
                        Err(_) => {
                            // Can't allocate, put back in queue (at front since already popped)
                            self.waiting_queue.push(entry);
                            break;
                        },
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Number of waiting requests
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Number of running requests
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of preempted requests
    pub fn preempted_count(&self) -> usize {
        self.preempted.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut output = SchedulerOutput::default();
        output.num_prefill_tokens = 100;
        output.num_decode_tokens = 10;
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
        let request_id = scheduler.add_request(vec![1, 2, 3], 10).unwrap();

        assert_eq!(request_id, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.stats().total_requests, 1);
    }

    #[test]
    fn test_scheduler_add_request_queue_full() {
        let mut scheduler = Scheduler::new(32, 1);
        let _ = scheduler.add_request(vec![1], 10).unwrap();

        let result = scheduler.add_request(vec![2], 10);
        assert!(matches!(result, Err(SchedulerError::QueueFull { .. })));
    }

    #[test]
    fn test_scheduler_schedule() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let _ = scheduler.add_request(vec![1, 2, 3], 10).unwrap();
        let output = scheduler.schedule(&mut kv_cache, 0).unwrap();

        assert_eq!(output.scheduled_request_ids.len(), 1);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_scheduler_update_after_iteration() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).unwrap();
        let _ = scheduler.schedule(&mut kv_cache, 0).unwrap();

        let mut generated = HashMap::new();
        generated.insert(request_id, 42u32);
        scheduler.update_after_iteration(&generated);

        let request = scheduler.get_request(request_id).unwrap();
        assert_eq!(request.generated_tokens, vec![42]);
        assert_eq!(request.iterations, 1);
    }

    #[test]
    fn test_scheduler_complete_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).unwrap();
        let _ = scheduler.schedule(&mut kv_cache, 0).unwrap();

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
            .unwrap();
        // Add high priority second
        let _high_id = scheduler
            .add_request_with_priority(vec![2], 10, Priority::High)
            .unwrap();

        // Schedule - should pick high priority
        let output = scheduler.schedule(&mut kv_cache, 0).unwrap();

        // With only 1 slot, we should see preemption or priority selection
        assert_eq!(output.scheduled_request_ids.len(), 1);

        // Low priority should still be waiting or preempted
        let low_request = scheduler.get_request(low_id).unwrap();
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
        let _ = scheduler.add_request(vec![1], 10).unwrap();
        let _ = scheduler.add_request(vec![2], 10).unwrap();
        let _ = scheduler.add_request(vec![3], 10).unwrap();

        let output = scheduler.schedule(&mut kv_cache, 0).unwrap();

        // Should only schedule 2 (max batch size)
        assert_eq!(output.scheduled_request_ids.len(), 2);
        assert_eq!(scheduler.running_count(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).unwrap();
        let _ = scheduler.schedule(&mut kv_cache, 0).unwrap();
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

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: SchedulerStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_requests, stats.total_requests);
        assert_eq!(parsed.preemptions, stats.preemptions);
    }
}
