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
// SLOT-BASED SERVER CONCURRENCY (per llama.cpp)
// ============================================================================
//
// llama.cpp uses a slot-based architecture for concurrent inference:
// - Fixed number of slots, each with its own KV cache allocation
// - State machine: IDLE → PROCESSING → GENERATING → (complete) → IDLE
// - Slots can be dynamically assigned to incoming requests
// - Enables efficient handling of multiple concurrent clients
// ============================================================================

/// Slot state machine (per llama.cpp server architecture)
///
/// Each slot transitions through these states:
/// - IDLE: Ready to accept new requests
/// - PROCESSING: Initial prompt processing (prefill phase)
/// - GENERATING: Token generation (decode phase)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotState {
    /// Slot is available for new requests
    Idle,
    /// Processing initial prompt (prefill)
    Processing,
    /// Generating tokens (decode)
    Generating,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Server slot for concurrent inference
///
/// Per llama.cpp: Each slot manages one inference request with its own
/// KV cache allocation and state machine.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Unique slot ID
    pub id: usize,
    /// Current state
    pub state: SlotState,
    /// Assigned request ID (None if idle)
    pub request_id: Option<u64>,
    /// Sequence ID for KV cache
    pub seq_id: Option<SeqId>,
    /// Input tokens (prompt)
    pub input_tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Number of prompt tokens processed
    pub n_prompt_tokens_processed: usize,
    /// Generation start time
    pub generation_start: Option<Instant>,
    /// Total prompt processing time (ms)
    pub prompt_time_ms: f64,
    /// Total generation time (ms)
    pub generation_time_ms: f64,
}

impl Slot {
    /// Create a new idle slot
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: SlotState::Idle,
            request_id: None,
            seq_id: None,
            input_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            max_tokens: 0,
            n_prompt_tokens_processed: 0,
            generation_start: None,
            prompt_time_ms: 0.0,
            generation_time_ms: 0.0,
        }
    }

    /// Check if slot is available
    pub fn is_idle(&self) -> bool {
        self.state == SlotState::Idle
    }

    /// Check if slot is actively generating
    pub fn is_generating(&self) -> bool {
        self.state == SlotState::Generating
    }

    /// Assign a request to this slot
    pub fn assign(&mut self, request_id: u64, input_tokens: Vec<u32>, max_tokens: usize) {
        self.state = SlotState::Processing;
        self.request_id = Some(request_id);
        self.input_tokens = input_tokens;
        self.max_tokens = max_tokens;
        self.generated_tokens.clear();
        self.n_prompt_tokens_processed = 0;
        self.prompt_time_ms = 0.0;
        self.generation_time_ms = 0.0;
        self.generation_start = None;
    }

    /// Transition from processing to generating
    pub fn start_generation(&mut self, prompt_time_ms: f64) {
        self.state = SlotState::Generating;
        self.prompt_time_ms = prompt_time_ms;
        self.generation_start = Some(Instant::now());
    }

    /// Add a generated token
    pub fn add_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }

    /// Check if generation is complete
    pub fn is_complete(&self, eos_token: u32) -> bool {
        if self.generated_tokens.len() >= self.max_tokens {
            return true;
        }
        if let Some(&last) = self.generated_tokens.last() {
            if last == eos_token {
                return true;
            }
        }
        false
    }

    /// Finish generation and reset to idle
    pub fn finish(&mut self) {
        if let Some(start) = self.generation_start {
            self.generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        self.state = SlotState::Idle;
        self.request_id = None;
        self.seq_id = None;
    }

    /// Get tokens per second for this slot's generation
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_time_ms > 0.0 {
            self.generated_tokens.len() as f64 / (self.generation_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Slot manager for concurrent inference
///
/// Per llama.cpp: Manages a fixed pool of slots for handling concurrent requests.
/// Each slot has its own KV cache allocation and can process one request at a time.
#[derive(Debug)]
pub struct SlotManager {
    /// Available slots
    slots: Vec<Slot>,
    /// Maximum context length per slot
    pub max_context_length: usize,
    /// Next request ID
    next_request_id: u64,
}

impl SlotManager {
    /// Create a new slot manager with the specified number of slots
    pub fn new(num_slots: usize, max_context_length: usize) -> Self {
        let slots = (0..num_slots).map(Slot::new).collect();
        Self {
            slots,
            max_context_length,
            next_request_id: 0,
        }
    }

    /// Get number of total slots
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Get number of idle slots
    pub fn num_idle_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_idle()).count()
    }

    /// Get number of active (non-idle) slots
    pub fn num_active_slots(&self) -> usize {
        self.slots.len() - self.num_idle_slots()
    }

    /// Find an idle slot
    pub fn find_idle_slot(&self) -> Option<usize> {
        self.slots.iter().position(Slot::is_idle)
    }

    /// Assign a request to an available slot
    ///
    /// Returns the slot ID if successful, None if no slots available.
    pub fn assign_request(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> Option<(usize, u64)> {
        let slot_id = self.find_idle_slot()?;
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        self.slots[slot_id].assign(request_id, input_tokens, max_tokens);
        Some((slot_id, request_id))
    }

    /// Get a reference to a slot
    pub fn get_slot(&self, slot_id: usize) -> Option<&Slot> {
        self.slots.get(slot_id)
    }

    /// Get a mutable reference to a slot
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Option<&mut Slot> {
        self.slots.get_mut(slot_id)
    }

    /// Get all active slots (non-idle)
    pub fn active_slots(&self) -> impl Iterator<Item = &Slot> {
        self.slots.iter().filter(|s| !s.is_idle())
    }

    /// Get all generating slots
    pub fn generating_slots(&self) -> impl Iterator<Item = &Slot> {
        self.slots.iter().filter(|s| s.is_generating())
    }

    /// Get slots ready for batch processing
    pub fn batch_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_generating())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get server utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.slots.is_empty() {
            0.0
        } else {
            self.num_active_slots() as f64 / self.slots.len() as f64
        }
    }

    /// Get aggregate tokens per second across all slots
    pub fn aggregate_tokens_per_second(&self) -> f64 {
        self.slots.iter().map(Slot::tokens_per_second).sum()
    }
}

// ============================================================================
// CONTINUOUS BATCHING (ubatch/sbatch per llama.cpp)
// ============================================================================
//
// llama.cpp's continuous batching system:
// - ubatch (micro-batch): Tokens processed in a single forward pass
// - sbatch (sequence batch): Multiple sequences grouped for batched inference
//
// This enables:
// - Dynamic batching: New sequences can join mid-inference
// - Efficient GPU utilization: Batch multiple decode steps together
// - Mixed prefill/decode: Process prefill and decode in same batch
// ============================================================================

/// Batch type for continuous batching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchType {
    /// Prefill batch (processing initial prompts)
    Prefill,
    /// Decode batch (generating tokens)
    Decode,
    /// Mixed prefill and decode
    Mixed,
}

impl Default for BatchType {
    fn default() -> Self {
        Self::Decode
    }
}

/// Token position within a batch
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BatchToken {
    /// Token ID
    pub token_id: u32,
    /// Sequence ID this token belongs to
    pub seq_idx: usize,
    /// Position within the sequence
    pub seq_pos: usize,
    /// Whether this is a prompt token (vs generated)
    pub is_prompt: bool,
}

impl BatchToken {
    /// Create a new batch token
    pub fn new(token_id: u32, seq_idx: usize, seq_pos: usize, is_prompt: bool) -> Self {
        Self {
            token_id,
            seq_idx,
            seq_pos,
            is_prompt,
        }
    }
}

/// Micro-batch (ubatch) - tokens for a single forward pass
///
/// Per llama.cpp: A ubatch contains tokens that will be processed together
/// in a single forward pass. Can contain tokens from multiple sequences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicroBatch {
    /// Tokens in this micro-batch
    pub tokens: Vec<BatchToken>,
    /// Sequence indices included in this batch
    pub seq_indices: Vec<usize>,
    /// Batch type (prefill/decode/mixed)
    pub batch_type: BatchType,
    /// Maximum sequence length in batch
    pub max_seq_len: usize,
    /// Number of prompt tokens in batch
    pub n_prompt_tokens: usize,
    /// Number of decode tokens in batch
    pub n_decode_tokens: usize,
}

impl MicroBatch {
    /// Create a new empty micro-batch
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            seq_indices: Vec::new(),
            batch_type: BatchType::Decode,
            max_seq_len: 0,
            n_prompt_tokens: 0,
            n_decode_tokens: 0,
        }
    }

    /// Create a micro-batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            seq_indices: Vec::new(),
            batch_type: BatchType::Decode,
            max_seq_len: 0,
            n_prompt_tokens: 0,
            n_decode_tokens: 0,
        }
    }

    /// Add a token to the batch
    pub fn add_token(&mut self, token: BatchToken) {
        if token.is_prompt {
            self.n_prompt_tokens += 1;
        } else {
            self.n_decode_tokens += 1;
        }

        // Track sequence index
        if !self.seq_indices.contains(&token.seq_idx) {
            self.seq_indices.push(token.seq_idx);
        }

        // Update max sequence length
        self.max_seq_len = self.max_seq_len.max(token.seq_pos + 1);

        self.tokens.push(token);

        // Update batch type
        self.update_batch_type();
    }

    /// Update batch type based on token composition
    fn update_batch_type(&mut self) {
        self.batch_type = match (self.n_prompt_tokens > 0, self.n_decode_tokens > 0) {
            (true, false) => BatchType::Prefill,
            (true, true) => BatchType::Mixed,
            // Both (false, true) and (false, false) result in Decode
            (false, _) => BatchType::Decode,
        };
    }

    /// Total number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of sequences in batch
    pub fn num_sequences(&self) -> usize {
        self.seq_indices.len()
    }

    /// Check if batch is pure prefill
    pub fn is_prefill(&self) -> bool {
        self.batch_type == BatchType::Prefill
    }

    /// Check if batch is pure decode
    pub fn is_decode(&self) -> bool {
        self.batch_type == BatchType::Decode
    }

    /// Check if batch is mixed
    pub fn is_mixed(&self) -> bool {
        self.batch_type == BatchType::Mixed
    }

    /// Get token IDs as a vector (for model input)
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.token_id).collect()
    }

    /// Get sequence positions (for position embeddings)
    pub fn positions(&self) -> Vec<usize> {
        self.tokens.iter().map(|t| t.seq_pos).collect()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.seq_indices.clear();
        self.batch_type = BatchType::Decode;
        self.max_seq_len = 0;
        self.n_prompt_tokens = 0;
        self.n_decode_tokens = 0;
    }
}

/// Sequence batch entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceBatchEntry {
    /// Sequence index
    pub seq_idx: usize,
    /// Slot ID
    pub slot_id: usize,
    /// Request ID
    pub request_id: u64,
    /// Current position in sequence
    pub position: usize,
    /// Tokens to process (for prefill)
    pub tokens: Vec<u32>,
    /// Is this sequence in prefill or decode mode
    pub is_prefill: bool,
}

impl SequenceBatchEntry {
    /// Create new sequence batch entry
    pub fn new(seq_idx: usize, slot_id: usize, request_id: u64) -> Self {
        Self {
            seq_idx,
            slot_id,
            request_id,
            position: 0,
            tokens: Vec::new(),
            is_prefill: true,
        }
    }

    /// Set tokens for prefill
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set position
    pub fn at_position(mut self, position: usize) -> Self {
        self.position = position;
        self
    }

    /// Mark as decode (not prefill)
    pub fn decoding(mut self) -> Self {
        self.is_prefill = false;
        self
    }
}

/// Sequence batch (sbatch) - multiple sequences for batched inference
///
/// Per llama.cpp: Groups sequences that will be processed together.
/// Manages the mapping from batch positions to individual sequences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SequenceBatch {
    /// Sequences in this batch
    pub sequences: Vec<SequenceBatchEntry>,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Current batch utilization
    pub utilization: f64,
}

impl SequenceBatch {
    /// Create a new sequence batch with max size
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            sequences: Vec::with_capacity(max_batch_size),
            max_batch_size,
            utilization: 0.0,
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(&mut self, entry: SequenceBatchEntry) -> bool {
        if self.sequences.len() >= self.max_batch_size {
            return false;
        }
        self.sequences.push(entry);
        self.update_utilization();
        true
    }

    /// Remove a sequence by index
    pub fn remove_sequence(&mut self, seq_idx: usize) -> Option<SequenceBatchEntry> {
        let pos = self.sequences.iter().position(|s| s.seq_idx == seq_idx)?;
        let entry = self.sequences.remove(pos);
        self.update_utilization();
        Some(entry)
    }

    /// Update utilization metric
    fn update_utilization(&mut self) {
        self.utilization = if self.max_batch_size > 0 {
            self.sequences.len() as f64 / self.max_batch_size as f64
        } else {
            0.0
        };
    }

    /// Number of sequences in batch
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.sequences.len() >= self.max_batch_size
    }

    /// Get prefill sequences
    pub fn prefill_sequences(&self) -> impl Iterator<Item = &SequenceBatchEntry> {
        self.sequences.iter().filter(|s| s.is_prefill)
    }

    /// Get decode sequences
    pub fn decode_sequences(&self) -> impl Iterator<Item = &SequenceBatchEntry> {
        self.sequences.iter().filter(|s| !s.is_prefill)
    }

    /// Count prefill sequences
    pub fn num_prefill(&self) -> usize {
        self.sequences.iter().filter(|s| s.is_prefill).count()
    }

    /// Count decode sequences
    pub fn num_decode(&self) -> usize {
        self.sequences.iter().filter(|s| !s.is_prefill).count()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.utilization = 0.0;
    }

    /// Get sequence by index
    pub fn get(&self, seq_idx: usize) -> Option<&SequenceBatchEntry> {
        self.sequences.iter().find(|s| s.seq_idx == seq_idx)
    }

    /// Get mutable sequence by index
    pub fn get_mut(&mut self, seq_idx: usize) -> Option<&mut SequenceBatchEntry> {
        self.sequences.iter_mut().find(|s| s.seq_idx == seq_idx)
    }
}

/// Batch configuration for continuous batching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum tokens per micro-batch
    pub max_ubatch_tokens: usize,
    /// Maximum sequences per sequence batch
    pub max_sbatch_sequences: usize,
    /// Prefer pure decode batches (vs mixed)
    pub prefer_pure_decode: bool,
    /// Maximum context length
    pub max_context_length: usize,
    /// Enable dynamic batching (add sequences mid-inference)
    pub dynamic_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_ubatch_tokens: 512,
            max_sbatch_sequences: 8,
            prefer_pure_decode: true,
            max_context_length: 2048,
            dynamic_batching: true,
        }
    }
}

impl BatchConfig {
    /// Create batch config with custom max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_ubatch_tokens = max_tokens;
        self
    }

    /// Create batch config with custom max sequences
    pub fn with_max_sequences(mut self, max_seqs: usize) -> Self {
        self.max_sbatch_sequences = max_seqs;
        self
    }
}

/// Batch scheduler for continuous batching
///
/// Coordinates micro-batch and sequence batch creation,
/// implementing llama.cpp-style continuous batching.
pub struct BatchScheduler {
    /// Configuration
    config: BatchConfig,
    /// Current sequence batch
    sbatch: SequenceBatch,
    /// Next sequence index
    next_seq_idx: usize,
    /// Statistics
    stats: BatchStats,
}

/// Statistics for batch scheduler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total micro-batches created
    pub ubatches_created: u64,
    /// Total sequence batches created
    pub sbatches_created: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Total prefill tokens
    pub prefill_tokens: u64,
    /// Total decode tokens
    pub decode_tokens: u64,
    /// Average tokens per ubatch
    pub avg_ubatch_size: f64,
    /// Average sequences per sbatch
    pub avg_sbatch_size: f64,
}

impl BatchScheduler {
    /// Create a new batch scheduler with default config
    pub fn new() -> Self {
        Self::with_config(BatchConfig::default())
    }

    /// Create a new batch scheduler with custom config
    pub fn with_config(config: BatchConfig) -> Self {
        let max_seqs = config.max_sbatch_sequences;
        Self {
            config,
            sbatch: SequenceBatch::new(max_seqs),
            next_seq_idx: 0,
            stats: BatchStats::default(),
        }
    }

    /// Add a new sequence to the batch scheduler
    pub fn add_sequence(
        &mut self,
        slot_id: usize,
        request_id: u64,
        input_tokens: Vec<u32>,
    ) -> Option<usize> {
        if self.sbatch.is_full() {
            return None;
        }

        let seq_idx = self.next_seq_idx;
        self.next_seq_idx += 1;

        let entry = SequenceBatchEntry::new(seq_idx, slot_id, request_id).with_tokens(input_tokens);

        if self.sbatch.add_sequence(entry) {
            Some(seq_idx)
        } else {
            None
        }
    }

    /// Mark sequence as completed and remove from batch
    pub fn complete_sequence(&mut self, seq_idx: usize) -> Option<SequenceBatchEntry> {
        self.sbatch.remove_sequence(seq_idx)
    }

    /// Transition sequence from prefill to decode
    pub fn start_decode(&mut self, seq_idx: usize, position: usize) -> bool {
        if let Some(entry) = self.sbatch.get_mut(seq_idx) {
            entry.is_prefill = false;
            entry.position = position;
            entry.tokens.clear(); // No longer need prefill tokens
            true
        } else {
            false
        }
    }

    /// Create a micro-batch from current sequences
    ///
    /// Returns a micro-batch optimized for the current state:
    /// - Prefill: Process all prompt tokens for prefill sequences
    /// - Decode: Process one token per decode sequence
    /// - Mixed: Combines both (if config allows)
    pub fn create_ubatch(&mut self) -> MicroBatch {
        let mut ubatch = MicroBatch::with_capacity(self.config.max_ubatch_tokens);

        // Process prefill sequences first (if any)
        for entry in &self.sbatch.sequences {
            if entry.is_prefill {
                // Add all prefill tokens
                for (i, &token_id) in entry.tokens.iter().enumerate() {
                    if ubatch.len() >= self.config.max_ubatch_tokens {
                        break;
                    }
                    ubatch.add_token(BatchToken::new(token_id, entry.seq_idx, i, true));
                }
            }
        }

        // If prefer_pure_decode and we have prefill tokens, return early
        if self.config.prefer_pure_decode && !ubatch.is_empty() && ubatch.is_prefill() {
            self.record_ubatch(&ubatch);
            return ubatch;
        }

        // Add decode tokens
        for entry in &self.sbatch.sequences {
            if !entry.is_prefill {
                if ubatch.len() >= self.config.max_ubatch_tokens {
                    break;
                }
                // Decode sequences process one token at their current position
                // (the actual token ID will be filled in during inference)
                ubatch.add_token(BatchToken::new(
                    0, // Placeholder - will be filled by inference
                    entry.seq_idx,
                    entry.position,
                    false,
                ));
            }
        }

        self.record_ubatch(&ubatch);
        ubatch
    }

    /// Record ubatch statistics
    fn record_ubatch(&mut self, ubatch: &MicroBatch) {
        if ubatch.is_empty() {
            return;
        }

        self.stats.ubatches_created += 1;
        self.stats.tokens_processed += ubatch.len() as u64;
        self.stats.prefill_tokens += ubatch.n_prompt_tokens as u64;
        self.stats.decode_tokens += ubatch.n_decode_tokens as u64;

        // Update rolling average
        let n = self.stats.ubatches_created as f64;
        self.stats.avg_ubatch_size =
            self.stats.avg_ubatch_size * (n - 1.0) / n + ubatch.len() as f64 / n;
    }

    /// Get the current sequence batch
    pub fn sbatch(&self) -> &SequenceBatch {
        &self.sbatch
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.sbatch.len()
    }

    /// Check if scheduler has capacity
    pub fn has_capacity(&self) -> bool {
        !self.sbatch.is_full()
    }

    /// Current batch utilization
    pub fn utilization(&self) -> f64 {
        self.sbatch.utilization
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DYNAMIC BATCH PRIORITY SCHEDULING
// ============================================================================
//
// Advanced priority scheduling with:
// - Age-based priority promotion (prevent starvation)
// - Deadline-aware scheduling (SLA support)
// - Priority-weighted token budgets
// - Multi-level feedback queue (MLFQ) style scheduling
// - Fair share allocation across priority levels
//
// Reference: Orca (Yu et al., 2022), vLLM priority scheduling
// ============================================================================

/// Deadline specification for a request
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Deadline {
    /// Target completion time (milliseconds from arrival)
    pub target_latency_ms: u64,
    /// Hard deadline (must complete by this time, else drop)
    pub hard_deadline_ms: Option<u64>,
    /// Soft SLA target (percentage of requests meeting target)
    pub sla_target: f64,
}

impl Default for Deadline {
    fn default() -> Self {
        Self {
            target_latency_ms: 1000, // 1 second default
            hard_deadline_ms: None,
            sla_target: 0.99, // 99% SLA
        }
    }
}

impl Deadline {
    /// Create a deadline with target latency
    pub fn with_target(target_ms: u64) -> Self {
        Self {
            target_latency_ms: target_ms,
            ..Default::default()
        }
    }

    /// Create a strict deadline with hard cutoff
    pub fn strict(target_ms: u64, hard_ms: u64) -> Self {
        Self {
            target_latency_ms: target_ms,
            hard_deadline_ms: Some(hard_ms),
            sla_target: 1.0,
        }
    }
}

/// Dynamic priority configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPriorityConfig {
    /// Enable age-based priority promotion
    pub enable_age_promotion: bool,
    /// Time (ms) before promoting to next priority level
    pub promotion_interval_ms: u64,
    /// Maximum priority level after promotion (prevent runaway)
    pub max_promoted_priority: Priority,
    /// Token budget per priority level (proportion of batch)
    pub priority_budgets: [f64; 4], // Low, Normal, High, Critical
    /// Enable deadline-aware scheduling
    pub enable_deadline_scheduling: bool,
    /// Urgency boost factor for approaching deadlines
    pub urgency_factor: f64,
    /// Minimum tokens to allocate per request
    pub min_tokens_per_request: usize,
    /// Enable fair share scheduling
    pub enable_fair_share: bool,
}

impl Default for DynamicPriorityConfig {
    fn default() -> Self {
        Self {
            enable_age_promotion: true,
            promotion_interval_ms: 5000, // Promote after 5 seconds
            max_promoted_priority: Priority::High,
            // Budget allocation: Low=5%, Normal=30%, High=40%, Critical=25%
            priority_budgets: [0.05, 0.30, 0.40, 0.25],
            enable_deadline_scheduling: true,
            urgency_factor: 2.0,
            min_tokens_per_request: 1,
            enable_fair_share: true,
        }
    }
}

impl DynamicPriorityConfig {
    /// Create config with custom budgets
    pub fn with_budgets(budgets: [f64; 4]) -> Self {
        Self {
            priority_budgets: budgets,
            ..Default::default()
        }
    }

    /// Disable age promotion
    pub fn no_promotion(mut self) -> Self {
        self.enable_age_promotion = false;
        self
    }

    /// Set promotion interval
    pub fn with_promotion_interval(mut self, ms: u64) -> Self {
        self.promotion_interval_ms = ms;
        self
    }
}

/// Request entry with dynamic priority tracking
#[derive(Debug, Clone)]
pub struct DynamicRequest {
    /// Base request data
    pub request_id: u64,
    /// Input tokens
    pub input_ids: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Original priority (as submitted)
    pub original_priority: Priority,
    /// Effective priority (may be promoted)
    pub effective_priority: Priority,
    /// Arrival time
    pub arrival_time: Instant,
    /// Deadline specification
    pub deadline: Option<Deadline>,
    /// Number of times priority was promoted
    pub promotions: u32,
    /// Current state
    pub state: SequenceState,
    /// Generated tokens
    pub generated_tokens: Vec<u32>,
    /// Sequence ID (when scheduled)
    pub seq_id: Option<SeqId>,
    /// Time-to-first-token (if started)
    pub ttft_ms: Option<f64>,
}

impl DynamicRequest {
    /// Create a new dynamic request
    pub fn new(request_id: u64, input_ids: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            request_id,
            input_ids,
            max_tokens,
            original_priority: Priority::Normal,
            effective_priority: Priority::Normal,
            arrival_time: Instant::now(),
            deadline: None,
            promotions: 0,
            state: SequenceState::Waiting,
            generated_tokens: Vec::new(),
            seq_id: None,
            ttft_ms: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.original_priority = priority;
        self.effective_priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Deadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Wait time since arrival
    pub fn wait_time_ms(&self) -> u64 {
        self.arrival_time.elapsed().as_millis() as u64
    }

    /// Check if deadline is approaching (within 2x target latency)
    pub fn is_urgent(&self) -> bool {
        if let Some(deadline) = &self.deadline {
            let elapsed = self.wait_time_ms();
            elapsed >= deadline.target_latency_ms / 2
        } else {
            false
        }
    }

    /// Check if hard deadline has passed
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = &self.deadline {
            if let Some(hard) = deadline.hard_deadline_ms {
                return self.wait_time_ms() > hard;
            }
        }
        false
    }

    /// Calculate urgency score (0.0 to 1.0+)
    /// Higher score = more urgent
    pub fn urgency_score(&self) -> f64 {
        if let Some(deadline) = &self.deadline {
            let elapsed = self.wait_time_ms() as f64;
            let target = deadline.target_latency_ms as f64;
            if target > 0.0 {
                elapsed / target
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Remaining tokens to generate
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.generated_tokens.len())
    }

    /// Total tokens (input + generated)
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len() + self.generated_tokens.len()
    }
}

/// Priority-aware entry for dynamic scheduling (reserved for heap-based scheduling)
#[derive(Debug)]
#[allow(dead_code)]
struct DynamicPriorityEntry {
    request_id: u64,
    effective_priority: Priority,
    urgency_score: f64,
    arrival_time: Instant,
}

impl PartialEq for DynamicPriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for DynamicPriorityEntry {}

impl PartialOrd for DynamicPriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DynamicPriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by: priority > urgency > arrival time (FIFO within same)
        match self.effective_priority.cmp(&other.effective_priority) {
            std::cmp::Ordering::Equal => {
                // Higher urgency first
                match self
                    .urgency_score
                    .partial_cmp(&other.urgency_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                {
                    std::cmp::Ordering::Equal => {
                        // Earlier arrival first (FIFO)
                        other.arrival_time.cmp(&self.arrival_time)
                    },
                    ord => ord,
                }
            },
            ord => ord,
        }
    }
}

/// Statistics for dynamic priority scheduling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DynamicSchedulerStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Requests completed
    pub completed_requests: u64,
    /// Requests that met SLA
    pub sla_met: u64,
    /// Requests that missed SLA
    pub sla_missed: u64,
    /// Requests dropped (hard deadline exceeded)
    pub dropped_requests: u64,
    /// Total priority promotions
    pub promotions: u64,
    /// Average time-to-first-token (ms)
    pub avg_ttft_ms: f64,
    /// p99 time-to-first-token (ms)
    pub p99_ttft_ms: f64,
    /// Tokens allocated per priority level
    pub tokens_by_priority: [u64; 4],
    /// Current queue depth per priority
    pub queue_depth_by_priority: [usize; 4],
}

/// Dynamic batch priority scheduler
///
/// Implements advanced priority scheduling with:
/// - Age-based priority promotion (MLFQ-style)
/// - Deadline-aware scheduling for SLA compliance
/// - Fair share token budget allocation
/// - Urgency-based boosting for time-sensitive requests
pub struct DynamicPriorityScheduler {
    /// Configuration
    config: DynamicPriorityConfig,
    /// All requests by ID
    requests: HashMap<u64, DynamicRequest>,
    /// Priority queues (one per level)
    priority_queues: [VecDeque<u64>; 4],
    /// Running requests
    running: Vec<u64>,
    /// Next request ID
    next_request_id: u64,
    /// Statistics
    stats: DynamicSchedulerStats,
    /// TTFT samples for percentile calculation
    ttft_samples: Vec<f64>,
    /// Total batch token budget
    batch_token_budget: usize,
}

impl DynamicPriorityScheduler {
    /// Create a new dynamic priority scheduler
    pub fn new(batch_token_budget: usize) -> Self {
        Self::with_config(batch_token_budget, DynamicPriorityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(batch_token_budget: usize, config: DynamicPriorityConfig) -> Self {
        Self {
            config,
            requests: HashMap::new(),
            priority_queues: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            running: Vec::new(),
            next_request_id: 0,
            stats: DynamicSchedulerStats::default(),
            ttft_samples: Vec::new(),
            batch_token_budget,
        }
    }

    /// Add a request with priority and optional deadline
    pub fn add_request(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        priority: Priority,
        deadline: Option<Deadline>,
    ) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let mut request =
            DynamicRequest::new(request_id, input_ids, max_tokens).with_priority(priority);
        if let Some(d) = deadline {
            request = request.with_deadline(d);
        }

        // Add to appropriate priority queue
        let queue_idx = priority as usize;
        self.priority_queues[queue_idx].push_back(request_id);
        self.requests.insert(request_id, request);

        self.stats.total_requests += 1;
        self.update_queue_depths();

        request_id
    }

    /// Add a simple request (Normal priority, no deadline)
    pub fn add_simple_request(&mut self, input_ids: Vec<u32>, max_tokens: usize) -> u64 {
        self.add_request(input_ids, max_tokens, Priority::Normal, None)
    }

    /// Perform age-based priority promotion
    pub fn promote_aged_requests(&mut self) {
        if !self.config.enable_age_promotion {
            return;
        }

        let promotion_threshold = self.config.promotion_interval_ms;
        let max_priority = self.config.max_promoted_priority;

        // Check each queue except Critical (can't promote beyond Critical)
        for queue_idx in 0..3 {
            let current_priority = match queue_idx {
                0 => Priority::Low,
                1 => Priority::Normal,
                2 => Priority::High,
                _ => continue,
            };

            // Skip if current priority is already at max promoted level
            if current_priority >= max_priority {
                continue;
            }

            // Find requests to promote
            let mut to_promote = Vec::new();
            for &request_id in &self.priority_queues[queue_idx] {
                if let Some(request) = self.requests.get(&request_id) {
                    let promotions_time = promotion_threshold * (request.promotions as u64 + 1);
                    if request.wait_time_ms() >= promotions_time {
                        to_promote.push(request_id);
                    }
                }
            }

            // Promote requests
            for request_id in to_promote {
                self.promote_request(request_id);
            }
        }
    }

    /// Promote a single request to next priority level
    fn promote_request(&mut self, request_id: u64) {
        if let Some(request) = self.requests.get_mut(&request_id) {
            let current_idx = request.effective_priority as usize;
            let max_idx = self.config.max_promoted_priority as usize;

            if current_idx < max_idx {
                // Remove from current queue
                self.priority_queues[current_idx].retain(|&id| id != request_id);

                // Promote
                let new_priority = match current_idx + 1 {
                    1 => Priority::Normal,
                    2 => Priority::High,
                    3 => Priority::Critical,
                    _ => return,
                };
                request.effective_priority = new_priority;
                request.promotions += 1;

                // Add to new queue (at front since it's been waiting)
                self.priority_queues[current_idx + 1].push_front(request_id);
                self.stats.promotions += 1;
            }
        }
    }

    /// Drop expired requests (hard deadline exceeded)
    pub fn drop_expired(&mut self) -> Vec<u64> {
        let mut dropped = Vec::new();

        for queue in &mut self.priority_queues {
            let mut to_remove = Vec::new();
            for &request_id in queue.iter() {
                if let Some(request) = self.requests.get(&request_id) {
                    if request.is_expired() {
                        to_remove.push(request_id);
                    }
                }
            }

            for request_id in to_remove {
                queue.retain(|&id| id != request_id);
                if let Some(mut request) = self.requests.remove(&request_id) {
                    request.state = SequenceState::Failed;
                    dropped.push(request_id);
                    self.stats.dropped_requests += 1;
                }
            }
        }

        self.update_queue_depths();
        dropped
    }

    /// Schedule requests using dynamic priority and token budgets
    ///
    /// Returns (scheduled_request_ids, tokens_allocated_per_request)
    pub fn schedule(&mut self, available_slots: usize) -> Vec<(u64, usize)> {
        // First, handle promotions and expirations
        self.promote_aged_requests();
        self.drop_expired();

        let mut scheduled = Vec::new();
        let mut remaining_budget = self.batch_token_budget;
        let mut remaining_slots = available_slots;

        // Calculate token budgets per priority level
        let budgets: [usize; 4] = if self.config.enable_fair_share {
            self.config
                .priority_budgets
                .map(|b| (b * self.batch_token_budget as f64) as usize)
        } else {
            [
                remaining_budget,
                remaining_budget,
                remaining_budget,
                remaining_budget,
            ]
        };

        // Schedule from highest priority to lowest
        for queue_idx in (0..4).rev() {
            if remaining_slots == 0 || remaining_budget == 0 {
                break;
            }

            let queue = &mut self.priority_queues[queue_idx];
            let mut priority_budget = budgets[queue_idx].min(remaining_budget);

            // Sort queue by urgency for deadline-aware scheduling
            if self.config.enable_deadline_scheduling {
                let mut sorted: Vec<_> = queue.iter().copied().collect();
                sorted.sort_by(|&a, &b| {
                    let req_a = self.requests.get(&a);
                    let req_b = self.requests.get(&b);
                    match (req_a, req_b) {
                        (Some(ra), Some(rb)) => rb
                            .urgency_score()
                            .partial_cmp(&ra.urgency_score())
                            .unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                *queue = sorted.into_iter().collect();
            }

            // Schedule requests from this priority level
            let mut scheduled_from_queue = Vec::new();
            for &request_id in queue.iter() {
                if remaining_slots == 0 || priority_budget < self.config.min_tokens_per_request {
                    break;
                }

                if let Some(request) = self.requests.get(&request_id) {
                    // Calculate tokens to allocate
                    let tokens_needed = request.remaining_tokens().max(1);
                    let tokens_to_allocate = tokens_needed
                        .min(priority_budget)
                        .max(self.config.min_tokens_per_request);

                    if tokens_to_allocate > 0 {
                        scheduled.push((request_id, tokens_to_allocate));
                        scheduled_from_queue.push(request_id);
                        priority_budget = priority_budget.saturating_sub(tokens_to_allocate);
                        remaining_budget = remaining_budget.saturating_sub(tokens_to_allocate);
                        remaining_slots -= 1;

                        // Track tokens by priority
                        self.stats.tokens_by_priority[queue_idx] += tokens_to_allocate as u64;
                    }
                }
            }

            // Remove scheduled requests from queue and update state
            for request_id in scheduled_from_queue {
                queue.retain(|&id| id != request_id);
                if let Some(request) = self.requests.get_mut(&request_id) {
                    request.state = SequenceState::Running;
                    self.running.push(request_id);

                    // Record TTFT if first time running
                    if request.ttft_ms.is_none() {
                        let ttft = request.wait_time_ms() as f64;
                        request.ttft_ms = Some(ttft);
                        self.ttft_samples.push(ttft);
                    }
                }
            }
        }

        self.update_queue_depths();
        scheduled
    }

    /// Complete a request and update statistics
    pub fn complete_request(&mut self, request_id: u64) -> Option<DynamicRequest> {
        // Remove from running
        self.running.retain(|&id| id != request_id);

        if let Some(mut request) = self.requests.remove(&request_id) {
            request.state = SequenceState::Completed;
            self.stats.completed_requests += 1;

            // Check SLA compliance
            if let Some(deadline) = &request.deadline {
                let elapsed = request.wait_time_ms();
                if elapsed <= deadline.target_latency_ms {
                    self.stats.sla_met += 1;
                } else {
                    self.stats.sla_missed += 1;
                }
            }

            // Update average TTFT
            self.update_ttft_stats();

            Some(request)
        } else {
            None
        }
    }

    /// Update TTFT statistics
    fn update_ttft_stats(&mut self) {
        if self.ttft_samples.is_empty() {
            return;
        }

        // Average
        let sum: f64 = self.ttft_samples.iter().sum();
        self.stats.avg_ttft_ms = sum / self.ttft_samples.len() as f64;

        // P99
        let mut sorted = self.ttft_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((sorted.len() as f64) * 0.99) as usize;
        self.stats.p99_ttft_ms = sorted
            .get(p99_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);
    }

    /// Update queue depth statistics
    fn update_queue_depths(&mut self) {
        for (i, queue) in self.priority_queues.iter().enumerate() {
            self.stats.queue_depth_by_priority[i] = queue.len();
        }
    }

    /// Get a request by ID
    pub fn get_request(&self, request_id: u64) -> Option<&DynamicRequest> {
        self.requests.get(&request_id)
    }

    /// Get statistics
    pub fn stats(&self) -> &DynamicSchedulerStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &DynamicPriorityConfig {
        &self.config
    }

    /// Total waiting requests
    pub fn waiting_count(&self) -> usize {
        self.priority_queues.iter().map(VecDeque::len).sum()
    }

    /// Running requests
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// SLA compliance rate (0.0 to 1.0)
    pub fn sla_compliance_rate(&self) -> f64 {
        let total = self.stats.sla_met + self.stats.sla_missed;
        if total == 0 {
            1.0
        } else {
            self.stats.sla_met as f64 / total as f64
        }
    }

    /// Get queue depth for a priority level
    pub fn queue_depth(&self, priority: Priority) -> usize {
        self.priority_queues[priority as usize].len()
    }
}

// ============================================================================
// CHUNKED PREFILL (per Sarathi-Serve / vLLM)
// ============================================================================
//
// Chunked prefill breaks long prompt processing into smaller chunks, allowing
// decode steps for other requests to interleave. This reduces TTFT variance
// and prevents head-of-line blocking from long prompts.
//
// Reference: Agrawal et al. (2024) "Sarathi-Serve: Large Language Model Inference
// with Chunked Prefill and Decode Disaggregation"

/// Configuration for chunked prefill
///
/// Controls how long prompts are split into chunks for interleaved processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedPrefillConfig {
    /// Enable chunked prefill (default: true)
    pub enabled: bool,
    /// Maximum tokens per prefill chunk (default: 512)
    pub chunk_size: usize,
    /// Minimum prompt length to trigger chunking (default: 256)
    pub min_prompt_length: usize,
    /// Allow decode interleaving between chunks (default: true)
    pub allow_decode_interleave: bool,
    /// Priority boost for partially-prefilled sequences (default: true)
    pub boost_partial_prefill: bool,
    /// Maximum chunks before forcing completion (default: 16)
    pub max_chunks: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chunk_size: 512,
            min_prompt_length: 256,
            allow_decode_interleave: true,
            boost_partial_prefill: true,
            max_chunks: 16,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Create config with specified chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Disable chunked prefill
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create config for low-latency scenarios (smaller chunks)
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            chunk_size: 128,
            min_prompt_length: 64,
            allow_decode_interleave: true,
            boost_partial_prefill: true,
            max_chunks: 32,
        }
    }

    /// Create config for high-throughput scenarios (larger chunks)
    pub fn high_throughput() -> Self {
        Self {
            enabled: true,
            chunk_size: 1024,
            min_prompt_length: 512,
            allow_decode_interleave: false,
            boost_partial_prefill: false,
            max_chunks: 8,
        }
    }
}

/// State of chunked prefill for a sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedPrefillState {
    /// Sequence ID
    pub seq_id: u64,
    /// Total prompt tokens
    pub total_tokens: usize,
    /// Tokens processed so far
    pub processed_tokens: usize,
    /// Current chunk index
    pub current_chunk: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Start time of prefill (for latency tracking)
    pub start_time_ms: u64,
    /// Time spent on each chunk (milliseconds)
    pub chunk_latencies: Vec<u64>,
}

impl ChunkedPrefillState {
    /// Create new chunked prefill state
    pub fn new(seq_id: u64, total_tokens: usize, chunk_size: usize) -> Self {
        let total_chunks = total_tokens.div_ceil(chunk_size);
        Self {
            seq_id,
            total_tokens,
            processed_tokens: 0,
            current_chunk: 0,
            total_chunks,
            start_time_ms: 0,
            chunk_latencies: Vec::with_capacity(total_chunks),
        }
    }

    /// Get next chunk of tokens to process
    pub fn next_chunk(&self, chunk_size: usize) -> std::ops::Range<usize> {
        let start = self.processed_tokens;
        let end = (start + chunk_size).min(self.total_tokens);
        start..end
    }

    /// Advance to next chunk
    pub fn advance(&mut self, tokens_processed: usize, latency_ms: u64) {
        self.processed_tokens += tokens_processed;
        self.current_chunk += 1;
        self.chunk_latencies.push(latency_ms);
    }

    /// Check if prefill is complete
    pub fn is_complete(&self) -> bool {
        self.processed_tokens >= self.total_tokens
    }

    /// Get progress as percentage
    pub fn progress(&self) -> f64 {
        if self.total_tokens == 0 {
            100.0
        } else {
            (self.processed_tokens as f64 / self.total_tokens as f64) * 100.0
        }
    }

    /// Remaining tokens to prefill
    pub fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.processed_tokens)
    }

    /// Average chunk latency
    pub fn avg_chunk_latency_ms(&self) -> f64 {
        if self.chunk_latencies.is_empty() {
            0.0
        } else {
            self.chunk_latencies.iter().sum::<u64>() as f64 / self.chunk_latencies.len() as f64
        }
    }
}

/// Statistics for chunked prefill scheduler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkedPrefillStats {
    /// Total sequences that used chunked prefill
    pub chunked_sequences: u64,
    /// Total sequences that bypassed chunking (short prompts)
    pub bypassed_sequences: u64,
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Total decode interleaves (decode steps between prefill chunks)
    pub decode_interleaves: u64,
    /// Sum of all chunk latencies (for averaging)
    pub total_chunk_latency_ms: u64,
    /// Maximum single chunk latency
    pub max_chunk_latency_ms: u64,
    /// Total tokens saved from prefix cache hits during chunked prefill
    pub prefix_cache_hits: u64,
}

impl ChunkedPrefillStats {
    /// Average chunk latency
    pub fn avg_chunk_latency_ms(&self) -> f64 {
        if self.chunks_processed == 0 {
            0.0
        } else {
            self.total_chunk_latency_ms as f64 / self.chunks_processed as f64
        }
    }

    /// Chunking rate (what % of sequences used chunking)
    pub fn chunking_rate(&self) -> f64 {
        let total = self.chunked_sequences + self.bypassed_sequences;
        if total == 0 {
            0.0
        } else {
            self.chunked_sequences as f64 / total as f64
        }
    }
}

/// Chunked prefill scheduler
///
/// Manages the chunked processing of long prompts, interleaving with decode
/// operations for improved TTFT across all requests.
pub struct ChunkedPrefillScheduler {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Active chunked prefill states (seq_id -> state)
    active_prefills: HashMap<u64, ChunkedPrefillState>,
    /// Queue of sequences waiting for prefill
    prefill_queue: VecDeque<u64>,
    /// Statistics
    stats: ChunkedPrefillStats,
    /// Next sequence ID
    next_seq_id: u64,
}

impl ChunkedPrefillScheduler {
    /// Create new chunked prefill scheduler
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            active_prefills: HashMap::new(),
            prefill_queue: VecDeque::new(),
            stats: ChunkedPrefillStats::default(),
            next_seq_id: 0,
        }
    }

    /// Submit a new sequence for prefill
    ///
    /// Returns the sequence ID and whether it will use chunked prefill.
    pub fn submit(&mut self, prompt_tokens: usize) -> (u64, bool) {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let use_chunking = self.config.enabled && prompt_tokens >= self.config.min_prompt_length;

        if use_chunking {
            let state = ChunkedPrefillState::new(seq_id, prompt_tokens, self.config.chunk_size);
            self.active_prefills.insert(seq_id, state);
            self.prefill_queue.push_back(seq_id);
            self.stats.chunked_sequences += 1;
        } else {
            self.stats.bypassed_sequences += 1;
        }

        (seq_id, use_chunking)
    }

    /// Get the next chunk to process
    ///
    /// Returns (seq_id, token_range) for the next prefill chunk, or None if
    /// all prefills are complete or queue is empty.
    pub fn next_chunk(&mut self) -> Option<(u64, std::ops::Range<usize>)> {
        // Find next sequence with remaining prefill work
        while let Some(&seq_id) = self.prefill_queue.front() {
            if let Some(state) = self.active_prefills.get(&seq_id) {
                if !state.is_complete() {
                    let range = state.next_chunk(self.config.chunk_size);
                    return Some((seq_id, range));
                }
            }
            // Remove completed or missing sequences from queue
            self.prefill_queue.pop_front();
        }
        None
    }

    /// Record completion of a prefill chunk
    pub fn complete_chunk(&mut self, seq_id: u64, tokens_processed: usize, latency_ms: u64) {
        if let Some(state) = self.active_prefills.get_mut(&seq_id) {
            state.advance(tokens_processed, latency_ms);
            self.stats.chunks_processed += 1;
            self.stats.total_chunk_latency_ms += latency_ms;
            self.stats.max_chunk_latency_ms = self.stats.max_chunk_latency_ms.max(latency_ms);

            // If complete, clean up
            if state.is_complete() {
                // Move to back of queue or remove
                if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
                    self.prefill_queue.remove(pos);
                }
            } else if self.config.boost_partial_prefill {
                // Keep at front for priority
            } else {
                // Move to back of queue for round-robin
                if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
                    self.prefill_queue.remove(pos);
                    self.prefill_queue.push_back(seq_id);
                }
            }
        }
    }

    /// Record a decode interleave (decode step between prefill chunks)
    pub fn record_decode_interleave(&mut self) {
        self.stats.decode_interleaves += 1;
    }

    /// Check if we should allow decode interleaving
    pub fn should_interleave_decode(&self) -> bool {
        self.config.allow_decode_interleave && !self.prefill_queue.is_empty()
    }

    /// Get state for a sequence
    pub fn get_state(&self, seq_id: u64) -> Option<&ChunkedPrefillState> {
        self.active_prefills.get(&seq_id)
    }

    /// Check if sequence has pending prefill work
    pub fn has_pending_prefill(&self, seq_id: u64) -> bool {
        self.active_prefills
            .get(&seq_id)
            .is_some_and(|s| !s.is_complete())
    }

    /// Remove completed sequence
    pub fn remove(&mut self, seq_id: u64) -> Option<ChunkedPrefillState> {
        if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
            self.prefill_queue.remove(pos);
        }
        self.active_prefills.remove(&seq_id)
    }

    /// Number of sequences with pending prefill
    pub fn pending_count(&self) -> usize {
        self.active_prefills
            .values()
            .filter(|s| !s.is_complete())
            .count()
    }

    /// Total prefill queue length
    pub fn queue_len(&self) -> usize {
        self.prefill_queue.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &ChunkedPrefillStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &ChunkedPrefillConfig {
        &self.config
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.active_prefills.clear();
        self.prefill_queue.clear();
    }

    /// Record prefix cache hit during prefill
    pub fn record_prefix_cache_hit(&mut self, tokens_saved: usize) {
        self.stats.prefix_cache_hits += tokens_saved as u64;
    }
}

impl Default for ChunkedPrefillScheduler {
    fn default() -> Self {
        Self::new(ChunkedPrefillConfig::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "heavy-tests"))]
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
}
