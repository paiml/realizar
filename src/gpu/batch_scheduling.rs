//! Batch Scheduling & Async Processing (PMAT-802)
//!
//! M25: Token Batching & Speculative Decoding
//! M26: Async I/O & Event-Driven Processing
//! M27: Request Scheduling & Resource Management



// =============================================================================
// M25: Token Batching & Speculative Decoding (Phase 16)
// =============================================================================

/// Token batch accumulator for batched processing (M25 - IMP-058)
///
/// Accumulates tokens until batch is full, then returns for processing.
/// Improves throughput by processing multiple tokens together.
#[derive(Debug)]
pub struct TokenBatch {
    tokens: Vec<usize>,
    capacity: usize,
}

impl TokenBatch {
    /// Create a new token batch with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get the batch capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current number of tokens in batch
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if batch is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Check if batch is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.tokens.len() >= self.capacity
    }

    /// Push a token to the batch
    ///
    /// Returns `Some(tokens)` when batch becomes full, `None` otherwise.
    pub fn push(&mut self, token: usize) -> Option<Vec<usize>> {
        self.tokens.push(token);
        if self.is_full() {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush and return all tokens, clearing the batch
    pub fn flush(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.tokens)
    }
}

/// Candidate token for speculative decoding
#[derive(Debug, Clone)]
struct SpeculativeCandidate {
    token: usize,
    /// Confidence score (stored for future use in acceptance thresholds)
    #[allow(dead_code)]
    confidence: f32,
}

/// Speculative token buffer for speculative decoding (M25 - IMP-059)
///
/// Manages candidate tokens generated speculatively, allowing verification
/// against actual model outputs for acceptance or rejection.
#[derive(Debug)]
pub struct SpeculativeBuffer {
    candidates: Vec<SpeculativeCandidate>,
    capacity: usize,
}

impl SpeculativeBuffer {
    /// Create a new speculative buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get the buffer capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current number of candidates
    #[must_use]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Add a candidate token with confidence score
    pub fn add_candidate(&mut self, token: usize, confidence: f32) {
        if self.candidates.len() < self.capacity {
            self.candidates
                .push(SpeculativeCandidate { token, confidence });
        }
    }

    /// Verify candidates against actual tokens
    ///
    /// Returns (num_accepted, rejection_index) where rejection_index is
    /// the first index where mismatch occurred, or None if all matched.
    #[must_use]
    pub fn verify(&self, actual_tokens: &[usize]) -> (usize, Option<usize>) {
        let mut accepted = 0;
        for (i, candidate) in self.candidates.iter().enumerate() {
            if i < actual_tokens.len() && candidate.token == actual_tokens[i] {
                accepted += 1;
            } else {
                return (accepted, Some(i));
            }
        }
        (accepted, None)
    }

    /// Accept first n candidates, removing them from buffer
    pub fn accept(&mut self, n: usize) {
        if n >= self.candidates.len() {
            self.candidates.clear();
        } else {
            self.candidates.drain(0..n);
        }
    }

    /// Reject all remaining candidates
    pub fn reject(&mut self) {
        self.candidates.clear();
    }
}

/// Batch ID for tracking inference batches
pub type BatchId = u64;

/// Inference batch scheduler for coordinating batched processing (M25 - IMP-060)
///
/// Manages pending and completed batches, allowing asynchronous batch
/// submission and result retrieval.
#[derive(Debug)]
pub struct InferenceBatchScheduler {
    next_id: BatchId,
    pending: std::collections::HashMap<BatchId, Vec<usize>>,
    completed: std::collections::VecDeque<(BatchId, Vec<usize>)>,
}

impl InferenceBatchScheduler {
    /// Create a new inference batch scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 0,
            pending: std::collections::HashMap::new(),
            completed: std::collections::VecDeque::new(),
        }
    }

    /// Get count of pending batches
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get count of completed batches
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Submit a batch for processing
    ///
    /// Returns a unique batch ID for tracking.
    pub fn submit(&mut self, tokens: Vec<usize>) -> BatchId {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(id, tokens);
        id
    }

    /// Mark a batch as complete with results
    pub fn complete(&mut self, batch_id: BatchId, results: Vec<usize>) {
        self.pending.remove(&batch_id);
        self.completed.push_back((batch_id, results));
    }

    /// Poll for a completed batch
    ///
    /// Returns `Some((batch_id, results))` if a batch is ready, `None` otherwise.
    pub fn poll(&mut self) -> Option<(BatchId, Vec<usize>)> {
        self.completed.pop_front()
    }

    /// Drain all completed batches
    pub fn drain(&mut self) -> Vec<(BatchId, Vec<usize>)> {
        self.completed.drain(..).collect()
    }
}

impl Default for InferenceBatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// M26: Async I/O & Event-Driven Processing (Phase 17)
// =============================================================================

/// Async request queue for non-blocking request handling (M26 - IMP-061)
///
/// Provides a bounded FIFO queue for inference requests with backpressure
/// support via try-based operations.
#[derive(Debug)]
pub struct AsyncRequestQueue<T> {
    items: std::collections::VecDeque<T>,
    capacity: usize,
}

impl<T> AsyncRequestQueue<T> {
    /// Create a new async request queue with specified capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            items: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Get queue capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current queue length
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Check if queue is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Try to push an item to the queue
    ///
    /// Returns `true` if successful, `false` if queue is full (backpressure).
    pub fn try_push(&mut self, item: T) -> bool {
        if self.is_full() {
            false
        } else {
            self.items.push_back(item);
            true
        }
    }

    /// Try to pop an item from the queue
    ///
    /// Returns `Some(item)` if available, `None` if queue is empty.
    pub fn try_pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }
}

/// Type alias for inference completion handler
pub type InferenceCompletionHandler = Box<dyn Fn(u64, &[usize]) + Send + Sync>;

/// Event notifier for inference completion (M26 - IMP-062)
///
/// Allows registration of handlers that are called when inference completes.
pub struct InferenceEventNotifier {
    handlers: Vec<InferenceCompletionHandler>,
}

impl std::fmt::Debug for InferenceEventNotifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEventNotifier")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl InferenceEventNotifier {
    /// Create a new event notifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Get count of registered handlers
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Register a completion handler
    ///
    /// Handler receives (request_id, output_tokens) when inference completes.
    pub fn register(&mut self, handler: InferenceCompletionHandler) {
        self.handlers.push(handler);
    }

    /// Notify all handlers of completion
    ///
    /// Calls each registered handler with the request ID and output tokens.
    pub fn notify(&self, request_id: u64, tokens: &[usize]) {
        for handler in &self.handlers {
            handler(request_id, tokens);
        }
    }

    /// Clear all registered handlers
    pub fn clear(&mut self) {
        self.handlers.clear();
    }
}

impl Default for InferenceEventNotifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Request ID type for timeout tracking
pub type RequestId = u64;

/// Timeout manager for request deadline tracking (M26 - IMP-063)
///
/// Tracks request deadlines and identifies expired requests.
#[derive(Debug)]
pub struct TimeoutManager {
    deadlines: std::collections::HashMap<RequestId, std::time::Instant>,
}

impl TimeoutManager {
    /// Create a new timeout manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            deadlines: std::collections::HashMap::new(),
        }
    }

    /// Get count of active timeout registrations
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.deadlines.len()
    }

    /// Register a timeout for a request
    ///
    /// The deadline is the absolute time at which the request should timeout.
    pub fn register(&mut self, request_id: RequestId, deadline: std::time::Instant) {
        self.deadlines.insert(request_id, deadline);
    }

    /// Remove timeout registration for a request
    ///
    /// Use when request completes before timeout.
    pub fn remove(&mut self, request_id: RequestId) {
        self.deadlines.remove(&request_id);
    }

    /// Check for expired requests and remove them
    ///
    /// Returns list of request IDs that have timed out.
    pub fn check_expired(&mut self) -> Vec<RequestId> {
        let now = std::time::Instant::now();
        let expired: Vec<RequestId> = self
            .deadlines
            .iter()
            .filter(|(_, &deadline)| now >= deadline)
            .map(|(&id, _)| id)
            .collect();

        for id in &expired {
            self.deadlines.remove(id);
        }

        expired
    }
}

impl Default for TimeoutManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// M27: Request Scheduling & Resource Management (Phase 18)
// =============================================================================

/// Priority level type (higher = more important)
pub type Priority = u32;

/// Priority request wrapper for priority queue (M27 - IMP-064)
#[derive(Debug, Clone)]
pub struct PriorityRequest<T> {
    priority: Priority,
    sequence: u64, // For FIFO ordering within same priority
    data: T,
}

impl<T> PriorityRequest<T> {
    /// Create a new priority request
    #[must_use]
    pub fn new(priority: Priority, data: T) -> Self {
        Self {
            priority,
            sequence: 0, // Will be set by queue
            data,
        }
    }

    /// Get the priority level
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Get reference to request data
    #[must_use]
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Consume and return the data
    #[must_use]
    pub fn into_data(self) -> T {
        self.data
    }
}

/// Priority request queue for request scheduling (M27 - IMP-064)
///
/// Implements priority-based scheduling with FIFO ordering for same-priority requests.
#[derive(Debug)]
pub struct PriorityRequestQueue<T> {
    items: Vec<PriorityRequest<T>>,
    next_sequence: u64,
}

impl<T> PriorityRequestQueue<T> {
    /// Create a new priority request queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Get number of items in queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Enqueue a request with priority
    pub fn enqueue(&mut self, mut request: PriorityRequest<T>) {
        request.sequence = self.next_sequence;
        self.next_sequence += 1;
        self.items.push(request);
    }

    /// Dequeue the highest priority request
    ///
    /// Returns the request with highest priority. For equal priorities,
    /// returns the earliest enqueued (FIFO).
    pub fn dequeue_highest(&mut self) -> Option<PriorityRequest<T>> {
        if self.items.is_empty() {
            return None;
        }

        // Find index of highest priority (and earliest sequence for ties)
        let mut best_idx = 0;
        for (i, item) in self.items.iter().enumerate().skip(1) {
            let best = &self.items[best_idx];
            if item.priority > best.priority
                || (item.priority == best.priority && item.sequence < best.sequence)
            {
                best_idx = i;
            }
        }

        Some(self.items.swap_remove(best_idx))
    }
}

impl<T> Default for PriorityRequestQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Token bucket rate limiter for throughput control (M27 - IMP-065)
///
/// Implements token bucket algorithm with configurable rate and burst capacity.
#[derive(Debug)]
pub struct TokenRateLimiter {
    tokens: u32,
    capacity: u32,
    rate: f64, // tokens per second
    last_refill: std::time::Instant,
}

impl TokenRateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `rate` - Tokens per second to refill
    /// * `burst_capacity` - Maximum tokens that can accumulate
    #[must_use]
    pub fn new(rate: f64, burst_capacity: u32) -> Self {
        Self {
            tokens: burst_capacity, // Start full
            capacity: burst_capacity,
            rate,
            last_refill: std::time::Instant::now(),
        }
    }

    /// Get current available tokens
    #[must_use]
    pub fn tokens_available(&self) -> u32 {
        self.tokens
    }

    /// Try to acquire tokens
    ///
    /// Returns `true` if tokens were acquired, `false` if insufficient tokens.
    pub fn try_acquire(&mut self, count: u32) -> bool {
        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    ///
    /// Call periodically to add tokens at the configured rate.
    pub fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = (elapsed * self.rate) as u32;

        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill = now;
        }
    }
}

/// Allocation ID for resource tracking
pub type AllocationId = u64;

/// Resource allocation record
#[derive(Debug, Clone)]
struct ResourceAllocation {
    memory: u64,
    compute: u32,
}

/// Resource usage tracker for memory and compute (M27 - IMP-066)
///
/// Tracks resource allocations and provides utilization metrics.
#[derive(Debug)]
pub struct ResourceTracker {
    memory_capacity: u64,
    compute_capacity: u32,
    memory_used: u64,
    compute_used: u32,
    allocations: std::collections::HashMap<AllocationId, ResourceAllocation>,
    next_id: AllocationId,
}

impl ResourceTracker {
    /// Create a new resource tracker
    ///
    /// # Arguments
    /// * `memory_capacity` - Total memory capacity in bytes
    /// * `compute_capacity` - Total compute capacity (0-100 percentage)
    #[must_use]
    pub fn new(memory_capacity: u64, compute_capacity: u32) -> Self {
        Self {
            memory_capacity,
            compute_capacity,
            memory_used: 0,
            compute_used: 0,
            allocations: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    /// Get current memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> u64 {
        self.memory_used
    }

    /// Get current compute usage (0-100)
    #[must_use]
    pub fn compute_usage(&self) -> u32 {
        self.compute_used
    }

    /// Check if allocation is possible
    #[must_use]
    pub fn can_allocate(&self, memory: u64, compute: u32) -> bool {
        self.memory_used + memory <= self.memory_capacity
            && self.compute_used + compute <= self.compute_capacity
    }

    /// Allocate resources
    ///
    /// Returns allocation ID if successful, None if insufficient resources.
    pub fn allocate(&mut self, memory: u64, compute: u32) -> Option<AllocationId> {
        if !self.can_allocate(memory, compute) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.memory_used += memory;
        self.compute_used += compute;
        self.allocations
            .insert(id, ResourceAllocation { memory, compute });

        Some(id)
    }

    /// Release allocated resources
    pub fn release(&mut self, id: AllocationId) {
        if let Some(alloc) = self.allocations.remove(&id) {
            self.memory_used = self.memory_used.saturating_sub(alloc.memory);
            self.compute_used = self.compute_used.saturating_sub(alloc.compute);
        }
    }

    /// Get usage as percentages
    ///
    /// Returns (memory_percentage, compute_percentage)
    #[must_use]
    pub fn usage_percentage(&self) -> (f64, f64) {
        let mem_pct = if self.memory_capacity > 0 {
            (self.memory_used as f64 / self.memory_capacity as f64) * 100.0
        } else {
            0.0
        };
        let compute_pct = if self.compute_capacity > 0 {
            (self.compute_used as f64 / self.compute_capacity as f64) * 100.0
        } else {
            0.0
        };
        (mem_pct, compute_pct)
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        // Default: 8GB memory, 100% compute
        Self::new(8 * 1024 * 1024 * 1024, 100)
    }
}
