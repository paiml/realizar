//! Chunked Prefill Scheduler Component
//!
//! Implements chunked prefill per Sarathi-Serve/vLLM for reduced TTFT variance.
//! Extracted from scheduler/mod.rs (PMAT-802).

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

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
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ChunkedPrefillConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = ChunkedPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.min_prompt_length, 256);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 16);
    }

    #[test]
    fn test_config_disabled() {
        let config = ChunkedPrefillConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_low_latency() {
        let config = ChunkedPrefillConfig::low_latency();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 128);
        assert_eq!(config.min_prompt_length, 64);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 32);
    }

    #[test]
    fn test_config_high_throughput() {
        let config = ChunkedPrefillConfig::high_throughput();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.min_prompt_length, 512);
        assert!(!config.allow_decode_interleave);
        assert!(!config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 8);
    }

    #[test]
    fn test_config_with_chunk_size() {
        let config = ChunkedPrefillConfig::default().with_chunk_size(256);
        assert_eq!(config.chunk_size, 256);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = ChunkedPrefillConfig::low_latency();
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: ChunkedPrefillConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config.enabled, restored.enabled);
        assert_eq!(config.chunk_size, restored.chunk_size);
        assert_eq!(config.min_prompt_length, restored.min_prompt_length);
    }

    // =========================================================================
    // ChunkedPrefillState Tests
    // =========================================================================

    #[test]
    fn test_state_new() {
        let state = ChunkedPrefillState::new(42, 1024, 256);
        assert_eq!(state.seq_id, 42);
        assert_eq!(state.total_tokens, 1024);
        assert_eq!(state.processed_tokens, 0);
        assert_eq!(state.current_chunk, 0);
        assert_eq!(state.total_chunks, 4); // 1024 / 256 = 4
        assert!(state.chunk_latencies.is_empty());
    }

    #[test]
    fn test_state_total_chunks_round_up() {
        // 1000 tokens / 256 = 3.9, should round up to 4
        let state = ChunkedPrefillState::new(1, 1000, 256);
        assert_eq!(state.total_chunks, 4);
    }

    #[test]
    fn test_state_next_chunk() {
        let state = ChunkedPrefillState::new(1, 1000, 256);
        let range = state.next_chunk(256);
        assert_eq!(range, 0..256);
    }

    #[test]
    fn test_state_next_chunk_last() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.processed_tokens = 768;
        let range = state.next_chunk(256);
        // Last chunk: 768..1000 (232 tokens, not 256)
        assert_eq!(range, 768..1000);
    }

    #[test]
    fn test_state_advance() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.advance(256, 50);
        assert_eq!(state.processed_tokens, 256);
        assert_eq!(state.current_chunk, 1);
        assert_eq!(state.chunk_latencies, vec![50]);
    }

    #[test]
    fn test_state_is_complete() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert!(!state.is_complete());
        state.processed_tokens = 1000;
        assert!(state.is_complete());
    }

    #[test]
    fn test_state_progress() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert!((state.progress() - 0.0).abs() < f64::EPSILON);
        state.processed_tokens = 500;
        assert!((state.progress() - 50.0).abs() < f64::EPSILON);
        state.processed_tokens = 1000;
        assert!((state.progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_progress_zero_tokens() {
        let state = ChunkedPrefillState::new(1, 0, 256);
        assert!((state.progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_remaining_tokens() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert_eq!(state.remaining_tokens(), 1000);
        state.processed_tokens = 600;
        assert_eq!(state.remaining_tokens(), 400);
        state.processed_tokens = 1000;
        assert_eq!(state.remaining_tokens(), 0);
    }

    #[test]
    fn test_state_avg_chunk_latency_empty() {
        let state = ChunkedPrefillState::new(1, 1000, 256);
        assert!((state.avg_chunk_latency_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_avg_chunk_latency() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.advance(256, 10);
        state.advance(256, 20);
        state.advance(256, 30);
        // Average: (10 + 20 + 30) / 3 = 20
        assert!((state.avg_chunk_latency_ms() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_serde_roundtrip() {
        let mut state = ChunkedPrefillState::new(42, 1000, 256);
        state.advance(256, 25);
        let json = serde_json::to_string(&state).expect("serialize");
        let restored: ChunkedPrefillState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(state.seq_id, restored.seq_id);
        assert_eq!(state.total_tokens, restored.total_tokens);
        assert_eq!(state.processed_tokens, restored.processed_tokens);
        assert_eq!(state.chunk_latencies, restored.chunk_latencies);
    }

    // =========================================================================
    // ChunkedPrefillStats Tests
    // =========================================================================

    #[test]
    fn test_stats_default() {
        let stats = ChunkedPrefillStats::default();
        assert_eq!(stats.chunked_sequences, 0);
        assert_eq!(stats.bypassed_sequences, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.decode_interleaves, 0);
        assert_eq!(stats.total_chunk_latency_ms, 0);
        assert_eq!(stats.max_chunk_latency_ms, 0);
        assert_eq!(stats.prefix_cache_hits, 0);
    }

    #[test]
    fn test_stats_avg_chunk_latency_zero() {
        let stats = ChunkedPrefillStats::default();
        assert!((stats.avg_chunk_latency_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_avg_chunk_latency() {
        let stats = ChunkedPrefillStats {
            chunks_processed: 4,
            total_chunk_latency_ms: 100,
            ..Default::default()
        };
        assert!((stats.avg_chunk_latency_ms() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_chunking_rate_zero() {
        let stats = ChunkedPrefillStats::default();
        assert!((stats.chunking_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_chunking_rate() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 3,
            bypassed_sequences: 1,
            ..Default::default()
        };
        // 3 / 4 = 0.75
        assert!((stats.chunking_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_serde_roundtrip() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 10,
            bypassed_sequences: 5,
            chunks_processed: 30,
            decode_interleaves: 20,
            total_chunk_latency_ms: 3000,
            max_chunk_latency_ms: 200,
            prefix_cache_hits: 100,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let restored: ChunkedPrefillStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(stats.chunked_sequences, restored.chunked_sequences);
        assert_eq!(stats.bypassed_sequences, restored.bypassed_sequences);
        assert_eq!(stats.chunks_processed, restored.chunks_processed);
    }

    // =========================================================================
    // ChunkedPrefillScheduler Tests
    // =========================================================================

    #[test]
    fn test_scheduler_new() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.config().enabled);
    }

    #[test]
    fn test_scheduler_submit_short_prompt() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // 100 tokens < 256 min_prompt_length, should bypass chunking
        let (seq_id, use_chunking) = scheduler.submit(100);
        assert_eq!(seq_id, 0);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_scheduler_submit_long_prompt() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // 1000 tokens >= 256 min_prompt_length, should use chunking
        let (seq_id, use_chunking) = scheduler.submit(1000);
        assert_eq!(seq_id, 0);
        assert!(use_chunking);
        assert_eq!(scheduler.stats().chunked_sequences, 1);
        assert_eq!(scheduler.stats().bypassed_sequences, 0);
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_submit_disabled() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::disabled());
        let (_, use_chunking) = scheduler.submit(1000);
        assert!(!use_chunking);
    }

    #[test]
    fn test_scheduler_next_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let result = scheduler.next_chunk();
        assert!(result.is_some());
        let (seq_id, range) = result.unwrap();
        assert_eq!(seq_id, 0);
        assert_eq!(range, 0..512); // Default chunk_size is 512
    }

    #[test]
    fn test_scheduler_next_chunk_empty() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.next_chunk().is_none());
    }

    #[test]
    fn test_scheduler_complete_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        scheduler.complete_chunk(0, 512, 100);

        assert_eq!(scheduler.stats().chunks_processed, 1);
        assert_eq!(scheduler.stats().total_chunk_latency_ms, 100);
        assert_eq!(scheduler.stats().max_chunk_latency_ms, 100);

        let state = scheduler.get_state(0).expect("state exists");
        assert_eq!(state.processed_tokens, 512);
    }

    #[test]
    fn test_scheduler_complete_all_chunks() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        // Complete in two chunks
        scheduler.complete_chunk(0, 512, 100);
        scheduler.complete_chunk(0, 488, 90);

        assert!(scheduler.get_state(0).unwrap().is_complete());
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_record_decode_interleave() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.record_decode_interleave();
        assert_eq!(scheduler.stats().decode_interleaves, 1);
    }

    #[test]
    fn test_scheduler_should_interleave_decode() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // No pending prefills, should not interleave
        assert!(!scheduler.should_interleave_decode());

        scheduler.submit(1000);
        // Now has pending prefills, should interleave
        assert!(scheduler.should_interleave_decode());
    }

    #[test]
    fn test_scheduler_should_interleave_disabled() {
        let config = ChunkedPrefillConfig {
            allow_decode_interleave: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);
        scheduler.submit(1000);
        assert!(!scheduler.should_interleave_decode());
    }

    #[test]
    fn test_scheduler_get_state() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let state = scheduler.get_state(0);
        assert!(state.is_some());
        assert_eq!(state.unwrap().total_tokens, 1000);

        let missing = scheduler.get_state(999);
        assert!(missing.is_none());
    }

    #[test]
    fn test_scheduler_has_pending_prefill() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        assert!(scheduler.has_pending_prefill(0));

        // Complete all chunks
        scheduler.complete_chunk(0, 512, 100);
        scheduler.complete_chunk(0, 488, 100);

        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_scheduler_remove() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let removed = scheduler.remove(0);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().seq_id, 0);
        assert!(scheduler.get_state(0).is_none());
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_clear() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);
        scheduler.submit(2000);

        scheduler.clear();

        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_record_prefix_cache_hit() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.record_prefix_cache_hit(100);
        assert_eq!(scheduler.stats().prefix_cache_hits, 100);

        scheduler.record_prefix_cache_hit(50);
        assert_eq!(scheduler.stats().prefix_cache_hits, 150);
    }

    #[test]
    fn test_scheduler_multiple_sequences() {
        let mut scheduler = ChunkedPrefillScheduler::default();

        let (id1, _) = scheduler.submit(1000);
        let (id2, _) = scheduler.submit(2000);
        let (id3, _) = scheduler.submit(500);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);

        assert_eq!(scheduler.stats().chunked_sequences, 3);
        assert_eq!(scheduler.pending_count(), 3);
    }
}
