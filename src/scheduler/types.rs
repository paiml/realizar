//! Scheduler Core Types
//!
//! Common types for the continuous batching scheduler.
//! Extracted from scheduler/mod.rs (PMAT-802).

use serde::{Deserialize, Serialize};

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
