//! Request Batching Infrastructure for GPU-Accelerated Inference
//!
//! Extracted from gguf_monolith.rs (PMAT-802) for vertical production partitioning.
//!
//! ## Contents
//!
//! - `BatchGenerationStats`: Statistics for batch generation capabilities
//! - `PendingRequest`, `RequestBatch`, `BatchRequestCollector`: Request batching
//! - `BatchingConfig`: Batching configuration
//! - `SlotState`, `ContinuousBatchScheduler`: Continuous batching
//! - `SpeculativeConfig`, `SpeculativeDecoder`: Speculative decoding
//! - `GpuBufferPool`, `GpuBufferPoolStats`: GPU buffer management
//! - `AsyncCommandQueue`, `CommandSlot`, `AsyncQueueStats`: Async command queue
//! - `PrefixCache`, `PrefixCacheEntry`, `PrefixCacheStats`: Prefix caching
//! - `MultiRequestState`, `MultiSchedulerRequest`, `SchedulingPolicy`, `MultiRequestScheduler`: Multi-request scheduling
//! - `ChunkedPrefillConfig`, `ChunkProgress`, `ChunkedPrefill`, `ChunkedPrefillStats`: Chunked prefill
//!
//! ## Feature Gate
//!
//! This entire module is gated behind `#[cfg(feature = "gpu")]`.

// Note: This module is feature-gated in mod.rs with #[cfg(feature = "gpu")]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

use super::runtime::OwnedQuantizedKVCache;

/// Statistics for batch generation configuration
#[derive(Debug, Clone)]
pub struct BatchGenerationStats {
    /// Whether GPU cache is ready
    pub gpu_cache_ready: bool,
    /// Memory used by GPU cache in GB
    pub cache_memory_gb: f64,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Recommended batch size for GPU efficiency
    pub recommended_batch_size: usize,
    /// Maximum batch size before memory pressure
    pub max_batch_size: usize,
}

// ============================================================================
// PARITY-023: Request Batching Infrastructure
// ============================================================================

/// A pending request waiting to be batched (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Request ID for tracking
    pub id: u64,
    /// Prompt tokens
    pub prompt: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Time when request was submitted
    pub submitted_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl PendingRequest {
    /// Create a new pending request
    pub fn new(
        id: u64,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            temperature,
            top_k,
            submitted_at: std::time::Instant::now(),
        }
    }

    /// Time spent waiting in queue
    pub fn wait_time(&self) -> std::time::Duration {
        self.submitted_at.elapsed()
    }
}

/// A batch of requests ready for processing (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct RequestBatch {
    /// Requests in this batch
    pub requests: Vec<PendingRequest>,
    /// When batch was formed
    pub formed_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl RequestBatch {
    /// Create batch from requests
    pub fn new(requests: Vec<PendingRequest>) -> Self {
        Self {
            requests,
            formed_at: std::time::Instant::now(),
        }
    }

    /// Number of requests in batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Extract prompts for batch processing
    pub fn prompts(&self) -> Vec<Vec<u32>> {
        self.requests.iter().map(|r| r.prompt.clone()).collect()
    }

    /// Average wait time for requests in this batch
    pub fn avg_wait_time(&self) -> std::time::Duration {
        if self.requests.is_empty() {
            return std::time::Duration::ZERO;
        }
        let total: std::time::Duration = self.requests.iter().map(PendingRequest::wait_time).sum();
        total / self.requests.len() as u32
    }
}

/// Request batch collector with configurable thresholds (PARITY-023)
///
/// Collects incoming requests and forms batches when:
/// - Batch size reaches `batch_threshold`, OR
/// - Wait time exceeds `timeout_ms`
///
/// This enables efficient GPU utilization by batching multiple requests.
#[cfg(feature = "gpu")]
pub struct BatchRequestCollector {
    /// Pending requests
    pending: std::sync::Mutex<Vec<PendingRequest>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
    /// Batch size threshold (32 = GPU GEMM threshold from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before forcing batch formation (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
}

#[cfg(feature = "gpu")]
impl BatchRequestCollector {
    /// Create new collector with default thresholds
    ///
    /// Default: batch_threshold=32, timeout_ms=50, max_batch_size=64
    pub fn new() -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
        }
    }

    /// Create collector with custom thresholds
    pub fn with_thresholds(batch_threshold: usize, timeout_ms: u64, max_batch_size: usize) -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold,
            timeout_ms,
            max_batch_size,
        }
    }

    /// Submit a request to the collector
    ///
    /// Returns the request ID for tracking
    pub fn submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = PendingRequest::new(id, prompt, max_tokens, temperature, top_k);

        let mut pending = self.pending.lock().expect("Mutex poisoned");
        pending.push(request);

        id
    }

    /// Check if batch is ready to be formed
    pub fn is_batch_ready(&self) -> bool {
        let pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return false;
        }

        // Batch ready if threshold reached
        if pending.len() >= self.batch_threshold {
            return true;
        }

        // Batch ready if oldest request has waited too long
        if let Some(oldest) = pending.first() {
            let wait_ms = oldest.wait_time().as_millis() as u64;
            if wait_ms >= self.timeout_ms {
                return true;
            }
        }

        false
    }

    /// Collect a batch of requests
    ///
    /// Returns None if no requests are pending or batch not ready
    pub fn collect_batch(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        // Check if batch is ready (threshold or timeout)
        let ready = pending.len() >= self.batch_threshold
            || pending
                .first()
                .is_some_and(|r| r.wait_time().as_millis() as u64 >= self.timeout_ms);

        if !ready {
            return None;
        }

        // Take up to max_batch_size requests
        let batch_size = pending.len().min(self.max_batch_size);
        let requests: Vec<PendingRequest> = pending.drain(..batch_size).collect();

        Some(RequestBatch::new(requests))
    }

    /// Force collect all pending requests as a batch
    pub fn flush(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        let requests: Vec<PendingRequest> = pending.drain(..).collect();
        Some(RequestBatch::new(requests))
    }

    /// Number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.lock().expect("Mutex poisoned").len()
    }

    /// Total requests submitted
    pub fn total_submitted(&self) -> u64 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "gpu")]
impl Default for BatchRequestCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Batching configuration for request collector (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Minimum batch size to trigger GPU processing (32 from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before processing smaller batch (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
    /// Whether to prefer latency (process immediately) or throughput (wait for batch)
    pub prefer_throughput: bool,
}

#[cfg(feature = "gpu")]
impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl BatchingConfig {
    /// Config optimized for latency (smaller batches, shorter timeout)
    pub fn latency_optimized() -> Self {
        Self {
            batch_threshold: 8,
            timeout_ms: 10,
            max_batch_size: 32,
            prefer_throughput: false,
        }
    }

    /// Config optimized for throughput (larger batches, longer timeout)
    pub fn throughput_optimized() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 100,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

/// Slot state for continuous batching (PARITY-028)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum SlotState {
    /// Slot is available for new request
    Empty,
    /// Slot has active request being generated
    Active {
        /// Unique request identifier
        request_id: u64,
        /// Input prompt tokens
        prompt_tokens: Vec<u32>,
        /// Tokens generated so far
        generated_tokens: Vec<u32>,
        /// Maximum tokens to generate
        max_tokens: usize,
        /// Sampling temperature
        temperature: f32,
        /// Top-k sampling parameter
        top_k: usize,
    },
    /// Slot has completed request waiting for retrieval
    Completed {
        /// Unique request identifier
        request_id: u64,
        /// All generated tokens
        generated_tokens: Vec<u32>,
    },
}

#[cfg(feature = "gpu")]
impl SlotState {
    /// Check if slot is available
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if slot has active generation
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    /// Check if slot has completed request
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Get request ID if slot has one
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Empty => None,
            Self::Active { request_id, .. } | Self::Completed { request_id, .. } => {
                Some(*request_id)
            },
        }
    }
}

/// Continuous batch scheduler (PARITY-028)
///
/// Enables dynamic addition/removal of requests from a running batch:
/// - Requests are assigned to slots
/// - Each slot can be in Empty, Active, or Completed state
/// - New requests fill empty slots immediately
/// - Completed requests free their slots for reuse
///
/// This maximizes GPU utilization by keeping the batch full.
#[cfg(feature = "gpu")]
pub struct ContinuousBatchScheduler {
    /// Fixed-size array of slots
    slots: std::sync::Mutex<Vec<SlotState>>,
    /// KV caches for each slot (pre-allocated)
    caches: std::sync::Mutex<Vec<OwnedQuantizedKVCache>>,
    /// Total slots (max concurrent requests)
    pub num_slots: usize,
    /// Completed request IDs for polling
    completed: std::sync::Mutex<Vec<(u64, Vec<u32>)>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl ContinuousBatchScheduler {
    /// Create scheduler with specified number of slots
    ///
    /// # Arguments
    /// * `num_slots` - Maximum concurrent requests (typically 32-64)
    /// * `num_layers` - Number of transformer layers (for KV cache)
    /// * `hidden_dim` - Hidden dimension (for KV cache)
    /// * `max_seq_len` - Maximum sequence length (for KV cache)
    pub fn new(num_slots: usize, num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let slots = vec![SlotState::Empty; num_slots];
        let caches = (0..num_slots)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len))
            .collect();

        Self {
            slots: std::sync::Mutex::new(slots),
            caches: std::sync::Mutex::new(caches),
            num_slots,
            completed: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a new request to the scheduler
    ///
    /// Returns request ID if slot available, None if all slots full
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Option<u64> {
        let mut slots = self.slots.lock().expect("Mutex poisoned");

        // Find first empty slot
        let empty_idx = slots.iter().position(SlotState::is_empty)?;

        let request_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        slots[empty_idx] = SlotState::Active {
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            temperature,
            top_k,
        };

        Some(request_id)
    }

    /// Get number of active slots
    pub fn active_count(&self) -> usize {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots.iter().filter(|s| s.is_active()).count()
    }

    /// Get number of empty slots
    pub fn empty_count(&self) -> usize {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots.iter().filter(|s| s.is_empty()).count()
    }

    /// Check if any slot has completed request
    pub fn has_completed(&self) -> bool {
        let completed = self.completed.lock().expect("Mutex poisoned");
        !completed.is_empty()
    }

    /// Retrieve completed request results
    pub fn poll_completed(&self) -> Vec<(u64, Vec<u32>)> {
        let mut completed = self.completed.lock().expect("Mutex poisoned");
        std::mem::take(&mut *completed)
    }

    /// Mark a request as completed and move to completed queue
    pub fn complete_request(&self, slot_idx: usize, tokens: Vec<u32>) {
        let mut slots = self.slots.lock().expect("Mutex poisoned");
        let mut completed = self.completed.lock().expect("Mutex poisoned");

        if slot_idx < slots.len() {
            if let SlotState::Active { request_id, .. } = &slots[slot_idx] {
                let id = *request_id;
                // Move to completed
                completed.push((id, tokens));
                // Free the slot
                slots[slot_idx] = SlotState::Empty;

                // Reset KV cache for this slot
                let mut caches = self.caches.lock().expect("Mutex poisoned");
                caches[slot_idx].reset();
            }
        }
    }

    /// Get active slot indices and their current positions
    pub fn get_active_slots(&self) -> Vec<(usize, usize)> {
        let slots = self.slots.lock().expect("Mutex poisoned");
        slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| match slot {
                SlotState::Active {
                    prompt_tokens,
                    generated_tokens,
                    ..
                } => {
                    let pos = prompt_tokens.len() + generated_tokens.len();
                    Some((idx, pos))
                },
                _ => None,
            })
            .collect()
    }

    /// Get utilization (active_slots / total_slots)
    pub fn utilization(&self) -> f64 {
        let active = self.active_count();
        active as f64 / self.num_slots as f64
    }
}

/// Speculative decoding configuration (PARITY-029)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculatively generate per step
    pub speculation_length: usize,
    /// Temperature for draft model (lower = more deterministic)
    pub draft_temperature: f32,
    /// Whether to use same model for draft (self-speculative)
    pub self_speculative: bool,
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: true,
        }
    }
}

/// Result of speculative decoding verification step
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted
    pub accepted_count: usize,
    /// Total draft tokens generated
    pub draft_count: usize,
    /// Accepted tokens (verified by target model)
    pub accepted_tokens: Vec<u32>,
    /// Whether all draft tokens were accepted
    pub all_accepted: bool,
}

/// Speculative decoder for accelerated token generation (PARITY-029)
///
/// Implements speculative decoding (Leviathan et al., 2023):
/// 1. Draft model generates K candidate tokens quickly
/// 2. Target model verifies all K tokens in parallel
/// 3. Accept tokens until first rejection, then resample
///
/// This enables O(K) speedup when draft acceptance rate is high.
#[cfg(feature = "gpu")]
pub struct SpeculativeDecoder {
    /// Speculative decoding configuration
    pub config: SpeculativeConfig,
    /// Statistics: total draft tokens generated
    pub total_draft_tokens: std::sync::atomic::AtomicU64,
    /// Statistics: total draft tokens accepted
    pub total_accepted_tokens: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl SpeculativeDecoder {
    /// Create new speculative decoder with default config
    pub fn new() -> Self {
        Self {
            config: SpeculativeConfig::default(),
            total_draft_tokens: std::sync::atomic::AtomicU64::new(0),
            total_accepted_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create speculative decoder with custom config
    pub fn with_config(config: SpeculativeConfig) -> Self {
        Self {
            config,
            total_draft_tokens: std::sync::atomic::AtomicU64::new(0),
            total_accepted_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get acceptance rate (accepted / total draft tokens)
    pub fn acceptance_rate(&self) -> f64 {
        let total = self
            .total_draft_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        let accepted = self
            .total_accepted_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        accepted as f64 / total as f64
    }

    /// Verify draft tokens against target model logits
    ///
    /// # Arguments
    /// * `draft_tokens` - Candidate tokens from draft model
    /// * `target_logits` - Logits from target model for each position
    /// * `temperature` - Sampling temperature for rejection sampling
    ///
    /// # Returns
    /// VerificationResult with accepted tokens and statistics
    pub fn verify_draft(
        &self,
        draft_tokens: &[u32],
        target_logits: &[Vec<f32>],
        temperature: f32,
    ) -> VerificationResult {
        let mut accepted_tokens = Vec::with_capacity(draft_tokens.len());
        let mut accepted_count = 0;

        // Verify each draft token against target model distribution
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() {
                break;
            }

            let logits = &target_logits[i];

            // Find target model's top token
            let (target_token, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // Accept if draft matches target (greedy case)
            if temperature == 0.0 {
                if draft_token == target_token as u32 {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject and use target's token instead
                    accepted_tokens.push(target_token as u32);
                    accepted_count += 1;
                    break; // Stop at first mismatch
                }
            } else {
                // Rejection sampling for non-greedy decoding
                // P(accept) = min(1, p_target(x) / p_draft(x))
                // For simplicity, accept if draft is in top-k of target
                let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
                sorted_indices.sort_by(|&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let top_k = 10; // Accept if in top-10
                let in_top_k = sorted_indices
                    .iter()
                    .take(top_k)
                    .any(|&idx| idx == draft_token as usize);

                if in_top_k {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject, use target's sampled token
                    accepted_tokens.push(sorted_indices[0] as u32);
                    accepted_count += 1;
                    break;
                }
            }
        }

        // Update statistics
        self.total_draft_tokens.fetch_add(
            draft_tokens.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.total_accepted_tokens
            .fetch_add(accepted_count as u64, std::sync::atomic::Ordering::Relaxed);

        VerificationResult {
            accepted_count,
            draft_count: draft_tokens.len(),
            accepted_tokens,
            all_accepted: accepted_count == draft_tokens.len(),
        }
    }

    /// Calculate expected speedup based on acceptance rate
    ///
    /// Speedup = K * acceptance_rate + 1 (always get at least 1 token)
    pub fn expected_speedup(&self) -> f64 {
        let k = self.config.speculation_length as f64;
        let acceptance_rate = self.acceptance_rate();
        k * acceptance_rate + 1.0
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.total_draft_tokens
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_accepted_tokens
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU Buffer Pool for zero-allocation inference (PARITY-031, IMP-309)
///
/// Pre-allocates GPU buffers during warmup to eliminate allocation overhead
/// during generation. Uses a pool of reusable buffers for each tensor type.
///
/// # Key Properties
/// - Zero GPU malloc after warmup phase
/// - Pre-allocated buffers for common tensor sizes
/// - Thread-safe buffer borrowing and return
///
/// # Buffer Types
/// - Hidden state buffers: [batch_size, hidden_dim]
/// - Intermediate buffers: [batch_size, intermediate_dim]
/// - Attention score buffers: [batch_size, num_heads, seq_len]
/// - KV cache buffers: [num_layers, seq_len, hidden_dim]
#[cfg(feature = "gpu")]
pub struct GpuBufferPool {
    /// Pre-allocated hidden state buffers
    hidden_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated intermediate buffers (FFN)
    intermediate_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated attention score buffers
    attention_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Buffer dimensions for validation
    hidden_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    num_heads: usize,
    /// Pool size per buffer type
    pool_size: usize,
    /// Statistics: buffers borrowed
    pub borrows: std::sync::atomic::AtomicU64,
    /// Statistics: buffers returned
    pub returns: std::sync::atomic::AtomicU64,
    /// Statistics: allocations after warmup (should be 0)
    pub post_warmup_allocs: std::sync::atomic::AtomicU64,
    /// Whether warmup is complete
    warmed_up: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "gpu")]
impl GpuBufferPool {
    /// Create new buffer pool with specified dimensions
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        max_seq_len: usize,
        num_heads: usize,
        pool_size: usize,
    ) -> Self {
        Self {
            hidden_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            intermediate_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            attention_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            hidden_dim,
            intermediate_dim,
            max_seq_len,
            num_heads,
            pool_size,
            borrows: std::sync::atomic::AtomicU64::new(0),
            returns: std::sync::atomic::AtomicU64::new(0),
            post_warmup_allocs: std::sync::atomic::AtomicU64::new(0),
            warmed_up: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Warmup: pre-allocate all buffers
    ///
    /// Call this once during model initialization to eliminate
    /// allocation overhead during inference.
    pub fn warmup(&self) {
        // Pre-allocate hidden state buffers
        {
            let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.hidden_dim]);
            }
        }

        // Pre-allocate intermediate buffers (FFN)
        {
            let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.intermediate_dim]);
            }
        }

        // Pre-allocate attention score buffers
        {
            let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
            for _ in 0..self.pool_size {
                buffers.push(vec![0.0f32; self.num_heads * self.max_seq_len]);
            }
        }

        self.warmed_up
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Borrow a hidden state buffer from the pool
    ///
    /// Returns a pre-allocated buffer if available, or allocates new if needed.
    pub fn borrow_hidden(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            // Need to allocate - track if after warmup
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.hidden_dim]
        }
    }

    /// Return a hidden state buffer to the pool
    pub fn return_hidden(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Zero out for security and determinism
        buffer.fill(0.0);

        let mut buffers = self.hidden_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Borrow an intermediate buffer from the pool
    pub fn borrow_intermediate(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.intermediate_dim]
        }
    }

    /// Return an intermediate buffer to the pool
    pub fn return_intermediate(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        buffer.fill(0.0);

        let mut buffers = self.intermediate_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
    }

    /// Borrow an attention score buffer from the pool
    pub fn borrow_attention(&self) -> Vec<f32> {
        self.borrows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                self.post_warmup_allocs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            vec![0.0f32; self.num_heads * self.max_seq_len]
        }
    }

    /// Return an attention score buffer to the pool
    pub fn return_attention(&self, mut buffer: Vec<f32>) {
        self.returns
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        buffer.fill(0.0);

        let mut buffers = self.attention_buffers.lock().expect("mutex poisoned");
        if buffers.len() < self.pool_size {
            buffers.push(buffer);
        }
    }

    /// Check if pool has achieved zero-allocation after warmup
    pub fn is_zero_alloc(&self) -> bool {
        self.warmed_up.load(std::sync::atomic::Ordering::Acquire)
            && self
                .post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
    }

    /// Get pool statistics
    pub fn stats(&self) -> GpuBufferPoolStats {
        GpuBufferPoolStats {
            borrows: self.borrows.load(std::sync::atomic::Ordering::Relaxed),
            returns: self.returns.load(std::sync::atomic::Ordering::Relaxed),
            post_warmup_allocs: self
                .post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed),
            warmed_up: self.warmed_up.load(std::sync::atomic::Ordering::Acquire),
            hidden_available: self.hidden_buffers.lock().expect("mutex poisoned").len(),
            intermediate_available: self
                .intermediate_buffers
                .lock()
                .expect("mutex poisoned")
                .len(),
            attention_available: self.attention_buffers.lock().expect("mutex poisoned").len(),
        }
    }

    /// Calculate total memory usage of the buffer pool
    pub fn memory_usage_bytes(&self) -> usize {
        let hidden_bytes = self.pool_size * self.hidden_dim * 4;
        let intermediate_bytes = self.pool_size * self.intermediate_dim * 4;
        let attention_bytes = self.pool_size * self.num_heads * self.max_seq_len * 4;
        hidden_bytes + intermediate_bytes + attention_bytes
    }
}

/// Statistics for GpuBufferPool
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuBufferPoolStats {
    /// Total borrows
    pub borrows: u64,
    /// Total returns
    pub returns: u64,
    /// Allocations after warmup (should be 0)
    pub post_warmup_allocs: u64,
    /// Whether warmup is complete
    pub warmed_up: bool,
    /// Available hidden buffers
    pub hidden_available: usize,
    /// Available intermediate buffers
    pub intermediate_available: usize,
    /// Available attention buffers
    pub attention_available: usize,
}

/// Async Command Queue for GPU pipelining (PARITY-032, IMP-310)
///
/// Implements double-buffering to hide GPU latency by overlapping
/// computation and data transfer. While one batch is being processed
/// on GPU, the next batch is being prepared on CPU.
///
/// # Key Properties
/// - Double-buffering: 2 command slots for overlap
/// - Async submission: Non-blocking command enqueue
/// - Pipeline stages: Prepare → Submit → Execute → Complete
///
/// # GPU Utilization Target
/// - Without pipelining: ~50% (waiting for results)
/// - With pipelining: >85% (overlapped execution)
#[cfg(feature = "gpu")]
pub struct AsyncCommandQueue {
    /// Command slots for double-buffering (2 slots)
    slots: [std::sync::Mutex<CommandSlot>; 2],
    /// Current slot index for submission
    current_slot: std::sync::atomic::AtomicUsize,
    /// Statistics: commands submitted
    pub commands_submitted: std::sync::atomic::AtomicU64,
    /// Statistics: commands completed
    pub commands_completed: std::sync::atomic::AtomicU64,
    /// Statistics: pipeline stalls (had to wait for previous)
    pub pipeline_stalls: std::sync::atomic::AtomicU64,
}

/// State of a command slot in the async queue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum CommandSlotState {
    /// Slot is empty and ready for new command
    Empty,
    /// Command is being prepared (CPU side)
    Preparing,
    /// Command has been submitted to GPU
    Submitted,
    /// Command execution is complete
    Complete,
}

/// A command slot for async execution
#[cfg(feature = "gpu")]
pub struct CommandSlot {
    /// Current state of this slot
    state: CommandSlotState,
    /// Input data for the command
    input: Option<Vec<f32>>,
    /// Output data from the command
    output: Option<Vec<f32>>,
    /// Timestamp when command was submitted
    submit_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl Default for CommandSlot {
    fn default() -> Self {
        Self {
            state: CommandSlotState::Empty,
            input: None,
            output: None,
            submit_time: None,
        }
    }
}

#[cfg(feature = "gpu")]
impl AsyncCommandQueue {
    /// Create new async command queue with double-buffering
    pub fn new() -> Self {
        Self {
            slots: [
                std::sync::Mutex::new(CommandSlot::default()),
                std::sync::Mutex::new(CommandSlot::default()),
            ],
            current_slot: std::sync::atomic::AtomicUsize::new(0),
            commands_submitted: std::sync::atomic::AtomicU64::new(0),
            commands_completed: std::sync::atomic::AtomicU64::new(0),
            pipeline_stalls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a command for async execution
    ///
    /// Returns the slot index where the command was placed.
    /// If both slots are busy, this will block until one is available
    /// (counted as a pipeline stall).
    pub fn submit(&self, input: Vec<f32>) -> usize {
        let slot_idx = self
            .current_slot
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            % 2;

        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");

        // Check if we need to wait for previous command
        if matches!(
            slot.state,
            CommandSlotState::Submitted | CommandSlotState::Preparing
        ) {
            self.pipeline_stalls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // In real implementation, would wait for GPU completion
            // For now, mark as complete to allow reuse
            slot.state = CommandSlotState::Complete;
        }

        // Prepare new command
        slot.state = CommandSlotState::Preparing;
        slot.input = Some(input);
        slot.output = None;
        slot.submit_time = Some(std::time::Instant::now());

        // Mark as submitted
        slot.state = CommandSlotState::Submitted;
        self.commands_submitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        slot_idx
    }

    /// Mark a command as complete with output
    pub fn complete(&self, slot_idx: usize, output: Vec<f32>) {
        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");
        slot.state = CommandSlotState::Complete;
        slot.output = Some(output);
        self.commands_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get output from a completed command
    ///
    /// Returns None if command is not complete yet.
    pub fn get_output(&self, slot_idx: usize) -> Option<Vec<f32>> {
        let mut slot = self.slots[slot_idx].lock().expect("mutex poisoned");
        if matches!(slot.state, CommandSlotState::Complete) {
            slot.state = CommandSlotState::Empty;
            slot.output.take()
        } else {
            None
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> AsyncQueueStats {
        let submitted = self
            .commands_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .commands_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let stalls = self
            .pipeline_stalls
            .load(std::sync::atomic::Ordering::Relaxed);

        // GPU utilization estimate: (1 - stalls/submitted) * 100
        let utilization = if submitted > 0 {
            (1.0 - stalls as f64 / submitted as f64) * 100.0
        } else {
            0.0
        };

        AsyncQueueStats {
            commands_submitted: submitted,
            commands_completed: completed,
            pipeline_stalls: stalls,
            in_flight: submitted.saturating_sub(completed),
            gpu_utilization_percent: utilization,
        }
    }

    /// Calculate pipeline efficiency
    ///
    /// Efficiency = commands without stall / total commands
    pub fn pipeline_efficiency(&self) -> f64 {
        let submitted = self
            .commands_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let stalls = self
            .pipeline_stalls
            .load(std::sync::atomic::Ordering::Relaxed);
        if submitted == 0 {
            return 1.0;
        }
        (submitted - stalls) as f64 / submitted as f64
    }
}

#[cfg(feature = "gpu")]
impl Default for AsyncCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for AsyncCommandQueue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct AsyncQueueStats {
    /// Total commands submitted
    pub commands_submitted: u64,
    /// Total commands completed
    pub commands_completed: u64,
    /// Pipeline stalls (had to wait)
    pub pipeline_stalls: u64,
    /// Commands currently in flight
    pub in_flight: u64,
    /// Estimated GPU utilization percentage
    pub gpu_utilization_percent: f64,
}

/// Prefix Cache for common prompts (PARITY-033, IMP-319)
///
/// Caches the KV cache state for common prompt prefixes, enabling
/// instant response (0ms TTFT) for repeated prompts.
///
/// # Key Properties
/// - Hash-based prefix lookup (FNV-1a)
/// - LRU eviction for memory management
/// - Thread-safe access
///
/// # Use Cases
/// - System prompts (cached once, reused for all requests)
/// - Common few-shot examples
/// - Chat history prefixes
#[cfg(feature = "gpu")]
pub struct PrefixCache {
    /// Cached prefix entries (hash → entry)
    entries: std::sync::Mutex<std::collections::HashMap<u64, PrefixCacheEntry>>,
    /// Maximum number of cached prefixes
    max_entries: usize,
    /// Statistics: cache hits
    pub hits: std::sync::atomic::AtomicU64,
    /// Statistics: cache misses
    pub misses: std::sync::atomic::AtomicU64,
    /// Statistics: evictions
    pub evictions: std::sync::atomic::AtomicU64,
}

/// A cached prefix entry
#[cfg(feature = "gpu")]
pub struct PrefixCacheEntry {
    /// The original prompt tokens
    pub tokens: Vec<u32>,
    /// Cached K state for each layer [num_layers, seq_len, hidden_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Cached V state for each layer [num_layers, seq_len, hidden_dim]
    pub v_cache: Vec<Vec<f32>>,
    /// Timestamp for LRU eviction
    pub last_access: std::time::Instant,
    /// Number of times this prefix was hit
    pub hit_count: u64,
}

#[cfg(feature = "gpu")]
impl PrefixCache {
    /// Create new prefix cache with specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(std::collections::HashMap::with_capacity(max_entries)),
            max_entries,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Hash tokens to create cache key (FNV-1a)
    fn hash_tokens(tokens: &[u32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET;
        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Look up a prefix in the cache
    ///
    /// Returns the cached KV state if found, None otherwise.
    #[allow(clippy::type_complexity)]
    pub fn lookup(&self, tokens: &[u32]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let hash = Self::hash_tokens(tokens);

        let mut entries = self.entries.lock().expect("mutex poisoned");
        if let Some(entry) = entries.get_mut(&hash) {
            // Verify tokens match (hash collision check)
            if entry.tokens == tokens {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                entry.last_access = std::time::Instant::now();
                entry.hit_count += 1;
                return Some((entry.k_cache.clone(), entry.v_cache.clone()));
            }
        }

        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    /// Insert a new prefix into the cache
    ///
    /// Evicts LRU entry if cache is full.
    pub fn insert(&self, tokens: Vec<u32>, k_cache: Vec<Vec<f32>>, v_cache: Vec<Vec<f32>>) {
        let hash = Self::hash_tokens(&tokens);

        let mut entries = self.entries.lock().expect("mutex poisoned");

        // Evict LRU if at capacity
        if entries.len() >= self.max_entries {
            // Find oldest entry
            if let Some((&oldest_hash, _)) = entries.iter().min_by_key(|(_, e)| e.last_access) {
                entries.remove(&oldest_hash);
                self.evictions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        entries.insert(
            hash,
            PrefixCacheEntry {
                tokens,
                k_cache,
                v_cache,
                last_access: std::time::Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Check if a prefix is cached
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let hash = Self::hash_tokens(tokens);
        let entries = self.entries.lock().expect("mutex poisoned");
        entries.contains_key(&hash)
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;

        PrefixCacheStats {
            hits,
            misses,
            evictions: self.evictions.load(std::sync::atomic::Ordering::Relaxed),
            entries: self.entries.lock().expect("mutex poisoned").len(),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut entries = self.entries.lock().expect("mutex poisoned");
        entries.clear();
    }

    /// Estimate memory usage of cached prefixes
    pub fn memory_usage_bytes(&self) -> usize {
        let entries = self.entries.lock().expect("mutex poisoned");
        entries
            .values()
            .map(|e| {
                let k_bytes: usize = e.k_cache.iter().map(|v| v.len() * 4).sum();
                let v_bytes: usize = e.v_cache.iter().map(|v| v.len() * 4).sum();
                let token_bytes = e.tokens.len() * 4;
                k_bytes + v_bytes + token_bytes
            })
            .sum()
    }
}

#[cfg(feature = "gpu")]
impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(16) // Default: cache 16 prefixes
    }
}

/// Statistics for PrefixCache
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evictions due to capacity
    pub evictions: u64,
    /// Current number of cached entries
    pub entries: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

// =============================================================================
// PARITY-034: Multi-Request Scheduler with Scheduling Policies (IMP-317)
// =============================================================================
//
// Extends PARITY-028's ContinuousBatchScheduler with:
// - Multiple scheduling policies (FCFS, SJF, Round-Robin)
// - Request queuing with priorities
// - TTFT (Time to First Token) tracking
// - Throughput scaling verification
//
// Architecture:
// - Incoming requests are queued with their KV cache states
// - Scheduler batches decode steps from multiple requests
// - GPU GEMM efficiency: batch_size > 1 enables GPU acceleration
// - Preemption: Long-running requests can be paused for new arrivals
// =============================================================================

/// Request state in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiRequestState {
    /// Waiting for prefill
    Pending,
    /// Prefill in progress
    Prefilling,
    /// Decode in progress
    Decoding,
    /// Request completed
    Completed,
    /// Request preempted (paused)
    Preempted,
}

/// A single inference request in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct MultiSchedulerRequest {
    /// Unique request ID
    pub id: u64,
    /// Input tokens
    pub tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Current state
    pub state: MultiRequestState,
    /// KV cache position (how many tokens processed)
    pub kv_position: usize,
    /// Arrival time for FCFS scheduling
    pub arrival_time: std::time::Instant,
    /// Time first token generated (for TTFT metric)
    pub first_token_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl MultiSchedulerRequest {
    /// Create new request
    pub fn new(id: u64, tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            tokens,
            generated: Vec::with_capacity(max_tokens),
            max_tokens,
            state: MultiRequestState::Pending,
            kv_position: 0,
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }
    }

    /// Check if request is complete
    pub fn is_complete(&self) -> bool {
        self.state == MultiRequestState::Completed || self.generated.len() >= self.max_tokens
    }

    /// Time to first token (None if not yet generated)
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time
            .map(|t| t.duration_since(self.arrival_time).as_secs_f64() * 1000.0)
    }
}

/// Scheduling policy for the batch scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    Fcfs,
    /// Shortest job first (by remaining tokens)
    Sjf,
    /// Round-robin with time slices
    RoundRobin,
}

/// Multi-request scheduler with scheduling policies (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestScheduler {
    /// Pending requests queue
    pending: std::sync::Mutex<std::collections::VecDeque<MultiSchedulerRequest>>,
    /// Active requests being processed
    active: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Completed requests
    completed: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum concurrent requests
    max_concurrent: usize,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Request ID counter
    next_id: std::sync::atomic::AtomicU64,
    /// Requests submitted
    pub requests_submitted: std::sync::atomic::AtomicU64,
    /// Requests completed
    pub requests_completed: std::sync::atomic::AtomicU64,
    /// Total tokens generated
    pub tokens_generated: std::sync::atomic::AtomicU64,
    /// Batch iterations performed
    pub batch_iterations: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl MultiRequestScheduler {
    /// Create new scheduler with given parameters
    pub fn new(max_batch_size: usize, max_concurrent: usize, policy: SchedulingPolicy) -> Self {
        Self {
            pending: std::sync::Mutex::new(std::collections::VecDeque::new()),
            active: std::sync::Mutex::new(Vec::with_capacity(max_concurrent)),
            completed: std::sync::Mutex::new(Vec::new()),
            max_batch_size,
            max_concurrent,
            policy,
            next_id: std::sync::atomic::AtomicU64::new(0),
            requests_submitted: std::sync::atomic::AtomicU64::new(0),
            requests_completed: std::sync::atomic::AtomicU64::new(0),
            tokens_generated: std::sync::atomic::AtomicU64::new(0),
            batch_iterations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a new request
    pub fn submit(&self, tokens: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = MultiSchedulerRequest::new(id, tokens, max_tokens);

        let mut pending = self.pending.lock().expect("mutex poisoned");
        pending.push_back(request);
        self.requests_submitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        id
    }

    /// Get batch of requests ready for decode step
    ///
    /// Returns request IDs and their current positions
    pub fn get_decode_batch(&self) -> Vec<(u64, usize)> {
        let mut active = self.active.lock().expect("mutex poisoned");
        let mut pending = self.pending.lock().expect("mutex poisoned");

        // Promote pending requests to active (up to max_concurrent)
        while active.len() < self.max_concurrent && !pending.is_empty() {
            if let Some(mut req) = pending.pop_front() {
                req.state = MultiRequestState::Decoding;
                active.push(req);
            }
        }

        // Sort by policy
        match self.policy {
            SchedulingPolicy::Fcfs => {
                // Already in arrival order
            },
            SchedulingPolicy::Sjf => {
                active.sort_by_key(|r| r.max_tokens - r.generated.len());
            },
            SchedulingPolicy::RoundRobin => {
                // Rotate - move first to end
                if active.len() > 1 {
                    let first = active.remove(0);
                    active.push(first);
                }
            },
        }

        // Return batch of decoding requests
        active
            .iter()
            .filter(|r| r.state == MultiRequestState::Decoding)
            .take(self.max_batch_size)
            .map(|r| (r.id, r.kv_position))
            .collect()
    }

    /// Record generated token for a request
    pub fn record_token(&self, request_id: u64, token: u32) {
        let mut active = self.active.lock().expect("mutex poisoned");

        if let Some(req) = active.iter_mut().find(|r| r.id == request_id) {
            // Record TTFT for first token
            if req.first_token_time.is_none() {
                req.first_token_time = Some(std::time::Instant::now());
            }

            req.generated.push(token);
            req.kv_position += 1;
            self.tokens_generated
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Check if complete
            if req.is_complete() {
                req.state = MultiRequestState::Completed;
            }
        }
    }

    /// Move completed requests from active to completed
    pub fn collect_completed(&self) -> Vec<MultiSchedulerRequest> {
        let mut active = self.active.lock().expect("mutex poisoned");
        let mut completed = self.completed.lock().expect("mutex poisoned");

        let (done, still_active): (Vec<_>, Vec<_>) = active
            .drain(..)
            .partition(|r| r.state == MultiRequestState::Completed);

        *active = still_active;

        for _req in &done {
            self.requests_completed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        completed.extend(done.iter().cloned());
        done
    }

    /// Run one batch iteration (for simulation)
    pub fn step(&self) {
        self.batch_iterations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> MultiRequestStats {
        let submitted = self
            .requests_submitted
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .requests_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let tokens = self
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let iterations = self
            .batch_iterations
            .load(std::sync::atomic::Ordering::Relaxed);

        let pending = self.pending.lock().expect("mutex poisoned").len();
        let active = self.active.lock().expect("mutex poisoned").len();

        MultiRequestStats {
            requests_submitted: submitted,
            requests_completed: completed,
            tokens_generated: tokens,
            batch_iterations: iterations,
            pending_requests: pending,
            active_requests: active,
            avg_batch_size: if iterations > 0 {
                tokens as f64 / iterations as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for multi-request scheduler (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestStats {
    /// Total requests submitted
    pub requests_submitted: u64,
    /// Total requests completed
    pub requests_completed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Batch iterations performed
    pub batch_iterations: u64,
    /// Current pending requests
    pub pending_requests: usize,
    /// Current active requests
    pub active_requests: usize,
    /// Average batch size
    pub avg_batch_size: f64,
}

// =============================================================================
// PARITY-035: Chunked Prefill for Long Contexts (IMP-320)
// =============================================================================
//
// Enables streaming prompt processing by breaking long prefills into chunks.
// Key optimization for TTFT (Time to First Token) with long contexts.
//
// Architecture:
// - Prompt is split into chunks (default 512 tokens)
// - Each chunk processes incrementally, updating KV cache
// - First token can be generated after first chunk completes
// - Total prefill time is spread across chunks
// =============================================================================

/// Configuration for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Chunk size in tokens (default: 512)
    pub chunk_size: usize,
    /// Maximum context length (default: 8192)
    pub max_context: usize,
    /// Whether to yield after each chunk for streaming
    pub stream_chunks: bool,
}

#[cfg(feature = "gpu")]
impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_context: 8192,
            stream_chunks: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl ChunkedPrefillConfig {
    /// Create config with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }
}

/// Progress report for a single chunk
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkProgress {
    /// Chunk index (0-based)
    pub chunk_idx: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Tokens processed so far
    pub tokens_processed: usize,
    /// Total tokens to process
    pub total_tokens: usize,
    /// Time for this chunk (ms)
    pub chunk_time_ms: f64,
    /// Cumulative time so far (ms)
    pub cumulative_time_ms: f64,
}

/// Chunked prefill processor for long context handling
#[cfg(feature = "gpu")]
pub struct ChunkedPrefill {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Chunks created from prompt
    chunks: Vec<Vec<u32>>,
    /// Current chunk being processed
    current_chunk: usize,
    /// Tokens processed so far
    tokens_processed: usize,
    /// Start time for timing
    start_time: Option<std::time::Instant>,
    /// Timing for each chunk
    chunk_times_ms: Vec<f64>,
}

#[cfg(feature = "gpu")]
impl ChunkedPrefill {
    /// Create new chunked prefill from prompt tokens
    pub fn new(prompt_tokens: &[u32], config: ChunkedPrefillConfig) -> Self {
        let chunks: Vec<Vec<u32>> = prompt_tokens
            .chunks(config.chunk_size)
            .map(<[u32]>::to_vec)
            .collect();

        Self {
            config,
            chunks,
            current_chunk: 0,
            tokens_processed: 0,
            start_time: None,
            chunk_times_ms: Vec::new(),
        }
    }

    /// Get total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Check if there are more chunks to process
    pub fn has_more_chunks(&self) -> bool {
        self.current_chunk < self.chunks.len()
    }

    /// Get the next chunk to process
    ///
    /// Returns None if all chunks are processed
    pub fn next_chunk(&mut self) -> Option<&[u32]> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        if self.current_chunk < self.chunks.len() {
            let chunk = &self.chunks[self.current_chunk];
            Some(chunk.as_slice())
        } else {
            None
        }
    }

    /// Mark current chunk as complete
    pub fn complete_chunk(&mut self, chunk_time_ms: f64) {
        if self.current_chunk < self.chunks.len() {
            self.tokens_processed += self.chunks[self.current_chunk].len();
            self.chunk_times_ms.push(chunk_time_ms);
            self.current_chunk += 1;
        }
    }

    /// Get progress after completing a chunk
    pub fn progress(&self) -> ChunkProgress {
        let cumulative_time_ms: f64 = self.chunk_times_ms.iter().sum();

        ChunkProgress {
            chunk_idx: self.current_chunk.saturating_sub(1),
            total_chunks: self.chunks.len(),
            tokens_processed: self.tokens_processed,
            total_tokens: self.total_tokens(),
            chunk_time_ms: self.chunk_times_ms.last().copied().unwrap_or(0.0),
            cumulative_time_ms,
        }
    }

    /// Get estimated time to first token (after first chunk)
    pub fn estimated_ttft_ms(&self) -> f64 {
        if let Some(first_chunk_time) = self.chunk_times_ms.first() {
            *first_chunk_time
        } else {
            // Estimate based on chunk size and typical throughput
            let tokens = self.chunks.first().map_or(0, Vec::len);
            // Conservative estimate: 0.5ms per token for prefill
            tokens as f64 * 0.5
        }
    }

    /// Get statistics after completion
    pub fn stats(&self) -> ChunkedPrefillStats {
        let total_time_ms: f64 = self.chunk_times_ms.iter().sum();
        let total_tokens = self.total_tokens();
        let avg_chunk_time_ms = if !self.chunk_times_ms.is_empty() {
            total_time_ms / self.chunk_times_ms.len() as f64
        } else {
            0.0
        };

        ChunkedPrefillStats {
            total_chunks: self.chunks.len(),
            chunk_size: self.config.chunk_size,
            total_tokens,
            total_time_ms,
            avg_chunk_time_ms,
            ttft_ms: self.estimated_ttft_ms(),
            tokens_per_second: if total_time_ms > 0.0 {
                total_tokens as f64 / (total_time_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Statistics for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillStats {
    /// Total chunks processed
    pub total_chunks: usize,
    /// Chunk size used
    pub chunk_size: usize,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Total time (ms)
    pub total_time_ms: f64,
    /// Average time per chunk (ms)
    pub avg_chunk_time_ms: f64,
    /// Time to first token (ms)
    pub ttft_ms: f64,
    /// Prefill throughput (tokens/sec)
    pub tokens_per_second: f64,
}


#[cfg(test)]
#[cfg(feature = "gpu")]
#[path = "batch_scheduler_tests.rs"]
mod batch_scheduler_tests;
