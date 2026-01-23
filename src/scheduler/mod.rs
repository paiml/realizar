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

// PMAT-802: Extracted modules
mod chunked_prefill;
mod types;
pub use chunked_prefill::{ChunkedPrefillConfig, ChunkedPrefillScheduler, ChunkedPrefillState, ChunkedPrefillStats};
pub use types::{Priority, SequenceState, SchedulerStats};

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
// Tests
// ============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod scheduler_tests;
