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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    // ==================== TokenBatch Tests ====================

    #[test]
    fn test_token_batch_new() {
        let batch = TokenBatch::new(8);
        assert_eq!(batch.capacity(), 8);
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_token_batch_push_partial() {
        let mut batch = TokenBatch::new(4);
        assert!(batch.push(1).is_none());
        assert!(batch.push(2).is_none());
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_token_batch_push_full() {
        let mut batch = TokenBatch::new(3);
        batch.push(1);
        batch.push(2);
        let result = batch.push(3);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
        assert!(batch.is_empty()); // Flushed
    }

    #[test]
    fn test_token_batch_flush() {
        let mut batch = TokenBatch::new(10);
        batch.push(10);
        batch.push(20);
        let flushed = batch.flush();
        assert_eq!(flushed, vec![10, 20]);
        assert!(batch.is_empty());
    }

    // ==================== SpeculativeBuffer Tests ====================

    #[test]
    fn test_speculative_buffer_new() {
        let buf = SpeculativeBuffer::new(5);
        assert_eq!(buf.capacity(), 5);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_speculative_buffer_add_candidate() {
        let mut buf = SpeculativeBuffer::new(3);
        buf.add_candidate(100, 0.9);
        buf.add_candidate(101, 0.8);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_speculative_buffer_capacity_limit() {
        let mut buf = SpeculativeBuffer::new(2);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);
        buf.add_candidate(3, 0.7); // Should be ignored
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_speculative_buffer_verify_all_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);
        buf.add_candidate(20, 0.8);
        buf.add_candidate(30, 0.7);

        let (accepted, rejection) = buf.verify(&[10, 20, 30]);
        assert_eq!(accepted, 3);
        assert!(rejection.is_none());
    }

    #[test]
    fn test_speculative_buffer_verify_partial_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);
        buf.add_candidate(20, 0.8);
        buf.add_candidate(30, 0.7);

        let (accepted, rejection) = buf.verify(&[10, 20, 99]);
        assert_eq!(accepted, 2);
        assert_eq!(rejection, Some(2));
    }

    #[test]
    fn test_speculative_buffer_verify_no_match() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(10, 0.9);

        let (accepted, rejection) = buf.verify(&[99]);
        assert_eq!(accepted, 0);
        assert_eq!(rejection, Some(0));
    }

    #[test]
    fn test_speculative_buffer_accept() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);
        buf.add_candidate(3, 0.7);

        buf.accept(2);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_speculative_buffer_accept_all() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);

        buf.accept(10); // More than available
        assert!(buf.is_empty());
    }

    #[test]
    fn test_speculative_buffer_reject() {
        let mut buf = SpeculativeBuffer::new(5);
        buf.add_candidate(1, 0.9);
        buf.add_candidate(2, 0.8);

        buf.reject();
        assert!(buf.is_empty());
    }

    // ==================== InferenceBatchScheduler Tests ====================

    #[test]
    fn test_inference_batch_scheduler_new() {
        let scheduler = InferenceBatchScheduler::new();
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.completed_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_default() {
        let scheduler = InferenceBatchScheduler::default();
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_submit() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id1 = scheduler.submit(vec![1, 2, 3]);
        let id2 = scheduler.submit(vec![4, 5]);

        assert_eq!(scheduler.pending_count(), 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_inference_batch_scheduler_complete() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id = scheduler.submit(vec![1, 2, 3]);
        assert_eq!(scheduler.pending_count(), 1);

        scheduler.complete(id, vec![100, 200]);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.completed_count(), 1);
    }

    #[test]
    fn test_inference_batch_scheduler_poll() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id = scheduler.submit(vec![1]);
        scheduler.complete(id, vec![99]);

        let result = scheduler.poll();
        assert!(result.is_some());
        let (batch_id, tokens) = result.unwrap();
        assert_eq!(batch_id, id);
        assert_eq!(tokens, vec![99]);
        assert_eq!(scheduler.completed_count(), 0);
    }

    #[test]
    fn test_inference_batch_scheduler_drain() {
        let mut scheduler = InferenceBatchScheduler::new();
        let id1 = scheduler.submit(vec![1]);
        let id2 = scheduler.submit(vec![2]);
        scheduler.complete(id1, vec![10]);
        scheduler.complete(id2, vec![20]);

        let drained = scheduler.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(scheduler.completed_count(), 0);
    }

    // ==================== AsyncRequestQueue Tests ====================

    #[test]
    fn test_async_request_queue_new() {
        let queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(5);
        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
    }

    #[test]
    fn test_async_request_queue_try_push() {
        let mut queue = AsyncRequestQueue::new(3);
        assert!(queue.try_push(1));
        assert!(queue.try_push(2));
        assert!(queue.try_push(3));
        assert!(!queue.try_push(4)); // Queue is full
        assert!(queue.is_full());
    }

    #[test]
    fn test_async_request_queue_try_pop() {
        let mut queue = AsyncRequestQueue::new(3);
        queue.try_push(10);
        queue.try_push(20);

        assert_eq!(queue.try_pop(), Some(10));
        assert_eq!(queue.try_pop(), Some(20));
        assert_eq!(queue.try_pop(), None);
    }

    #[test]
    fn test_async_request_queue_fifo_order() {
        let mut queue = AsyncRequestQueue::new(5);
        for i in 0..5 {
            queue.try_push(i);
        }
        for i in 0..5 {
            assert_eq!(queue.try_pop(), Some(i));
        }
    }

    // ==================== InferenceEventNotifier Tests ====================

    #[test]
    fn test_inference_event_notifier_new() {
        let notifier = InferenceEventNotifier::new();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_default() {
        let notifier = InferenceEventNotifier::default();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_register() {
        let mut notifier = InferenceEventNotifier::new();
        notifier.register(Box::new(|_id, _tokens| {}));
        notifier.register(Box::new(|_id, _tokens| {}));
        assert_eq!(notifier.handler_count(), 2);
    }

    #[test]
    fn test_inference_event_notifier_notify() {
        let mut notifier = InferenceEventNotifier::new();
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        notifier.register(Box::new(move |id, _tokens| {
            counter_clone.fetch_add(id, Ordering::SeqCst);
        }));

        notifier.notify(42, &[1, 2, 3]);
        assert_eq!(counter.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_inference_event_notifier_clear() {
        let mut notifier = InferenceEventNotifier::new();
        notifier.register(Box::new(|_id, _tokens| {}));
        notifier.clear();
        assert_eq!(notifier.handler_count(), 0);
    }

    #[test]
    fn test_inference_event_notifier_debug() {
        let notifier = InferenceEventNotifier::new();
        let debug_str = format!("{:?}", notifier);
        assert!(debug_str.contains("handler_count"));
    }

    // ==================== TimeoutManager Tests ====================

    #[test]
    fn test_timeout_manager_new() {
        let manager = TimeoutManager::new();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_default() {
        let manager = TimeoutManager::default();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_register() {
        let mut manager = TimeoutManager::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        manager.register(1, deadline);
        manager.register(2, deadline);
        assert_eq!(manager.active_count(), 2);
    }

    #[test]
    fn test_timeout_manager_remove() {
        let mut manager = TimeoutManager::new();
        manager.register(1, Instant::now() + Duration::from_secs(10));
        manager.remove(1);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_timeout_manager_check_expired() {
        let mut manager = TimeoutManager::new();
        // Already expired
        manager.register(1, Instant::now() - Duration::from_secs(1));
        // Not expired
        manager.register(2, Instant::now() + Duration::from_secs(60));

        let expired = manager.check_expired();
        assert_eq!(expired, vec![1]);
        assert_eq!(manager.active_count(), 1); // Only non-expired remains
    }

    // ==================== PriorityRequest Tests ====================

    #[test]
    fn test_priority_request_new() {
        let req = PriorityRequest::new(5, "data");
        assert_eq!(req.priority(), 5);
        assert_eq!(req.data(), &"data");
    }

    #[test]
    fn test_priority_request_into_data() {
        let req = PriorityRequest::new(10, vec![1, 2, 3]);
        let data = req.into_data();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_priority_request_clone() {
        let req = PriorityRequest::new(3, "test");
        let cloned = req.clone();
        assert_eq!(cloned.priority(), 3);
    }

    // ==================== PriorityRequestQueue Tests ====================

    #[test]
    fn test_priority_request_queue_new() {
        let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_request_queue_default() {
        let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::default();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_request_queue_enqueue() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(1, "low"));
        queue.enqueue(PriorityRequest::new(5, "high"));
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_priority_request_queue_dequeue_highest() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(1, "low"));
        queue.enqueue(PriorityRequest::new(10, "high"));
        queue.enqueue(PriorityRequest::new(5, "medium"));

        let top = queue.dequeue_highest().unwrap();
        assert_eq!(top.priority(), 10);
        assert_eq!(top.data(), &"high");
    }

    #[test]
    fn test_priority_request_queue_fifo_same_priority() {
        let mut queue = PriorityRequestQueue::new();
        queue.enqueue(PriorityRequest::new(5, "first"));
        queue.enqueue(PriorityRequest::new(5, "second"));
        queue.enqueue(PriorityRequest::new(5, "third"));

        // Should return in FIFO order for same priority
        assert_eq!(queue.dequeue_highest().unwrap().data(), &"first");
        assert_eq!(queue.dequeue_highest().unwrap().data(), &"second");
        assert_eq!(queue.dequeue_highest().unwrap().data(), &"third");
    }

    #[test]
    fn test_priority_request_queue_empty() {
        let mut queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
        assert!(queue.dequeue_highest().is_none());
    }

    // ==================== TokenRateLimiter Tests ====================

    #[test]
    fn test_token_rate_limiter_new() {
        let limiter = TokenRateLimiter::new(10.0, 100);
        assert_eq!(limiter.tokens_available(), 100); // Starts full
    }

    #[test]
    fn test_token_rate_limiter_try_acquire() {
        let mut limiter = TokenRateLimiter::new(10.0, 50);
        assert!(limiter.try_acquire(30));
        assert_eq!(limiter.tokens_available(), 20);
        assert!(limiter.try_acquire(20));
        assert_eq!(limiter.tokens_available(), 0);
        assert!(!limiter.try_acquire(1));
    }

    #[test]
    fn test_token_rate_limiter_refill() {
        let mut limiter = TokenRateLimiter::new(1000.0, 100);
        limiter.try_acquire(100);
        assert_eq!(limiter.tokens_available(), 0);

        // Sleep briefly to allow some refill
        std::thread::sleep(Duration::from_millis(50));
        limiter.refill();
        assert!(limiter.tokens_available() > 0);
    }

    #[test]
    fn test_token_rate_limiter_capacity_limit() {
        let mut limiter = TokenRateLimiter::new(1000.0, 10);
        // Already full
        std::thread::sleep(Duration::from_millis(50));
        limiter.refill();
        // Should not exceed capacity
        assert!(limiter.tokens_available() <= 10);
    }

    // ==================== ResourceTracker Tests ====================

    #[test]
    fn test_resource_tracker_new() {
        let tracker = ResourceTracker::new(1024, 100);
        assert_eq!(tracker.memory_usage(), 0);
        assert_eq!(tracker.compute_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_default() {
        let tracker = ResourceTracker::default();
        assert_eq!(tracker.memory_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_can_allocate() {
        let tracker = ResourceTracker::new(1000, 100);
        assert!(tracker.can_allocate(500, 50));
        assert!(!tracker.can_allocate(2000, 50));
        assert!(!tracker.can_allocate(500, 150));
    }

    #[test]
    fn test_resource_tracker_allocate() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id = tracker.allocate(300, 30);
        assert!(id.is_some());
        assert_eq!(tracker.memory_usage(), 300);
        assert_eq!(tracker.compute_usage(), 30);
    }

    #[test]
    fn test_resource_tracker_allocate_failure() {
        let mut tracker = ResourceTracker::new(100, 100);
        let id = tracker.allocate(200, 50);
        assert!(id.is_none());
    }

    #[test]
    fn test_resource_tracker_release() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id = tracker.allocate(500, 50).unwrap();
        tracker.release(id);
        assert_eq!(tracker.memory_usage(), 0);
        assert_eq!(tracker.compute_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_multiple_allocations() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id1 = tracker.allocate(200, 20).unwrap();
        let id2 = tracker.allocate(300, 30).unwrap();

        assert_eq!(tracker.memory_usage(), 500);
        assert_eq!(tracker.compute_usage(), 50);

        tracker.release(id1);
        assert_eq!(tracker.memory_usage(), 300);
        assert_eq!(tracker.compute_usage(), 30);

        tracker.release(id2);
        assert_eq!(tracker.memory_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_usage_percentage() {
        let mut tracker = ResourceTracker::new(1000, 100);
        tracker.allocate(500, 25).unwrap();

        let (mem_pct, compute_pct) = tracker.usage_percentage();
        assert!((mem_pct - 50.0).abs() < 0.1);
        assert!((compute_pct - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_resource_tracker_usage_percentage_zero_capacity() {
        let tracker = ResourceTracker::new(0, 0);
        let (mem_pct, compute_pct) = tracker.usage_percentage();
        assert!((mem_pct - 0.0).abs() < 0.1);
        assert!((compute_pct - 0.0).abs() < 0.1);
    }
}
