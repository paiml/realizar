//! Mixture-of-Experts (MOE) routing with Capacity Factor load balancing
//!
//! Implements inference-time load balancing per Fedus et al. (2022) Switch Transformers.
//!
//! ## Features
//!
//! - **Power of Two Choices**: Mitzenmacher (2001) load balancing algorithm
//! - **Capacity Factor Routing**: Fedus et al. (2022) expert capacity limits
//! - **Circuit Breaker**: Nygard (2018) failure isolation pattern
//! - **Heijunka Controller**: Toyota Production System load leveling via Little's Law
//! - **Andon Triggers**: Jidoka (built-in quality) automated quality control

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use crate::error::{RealizarError, Result};

/// Configuration for capacity factor routing
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Maximum queue depth per expert
    pub capacity: usize,
    /// Number of experts
    pub num_experts: usize,
}

/// Capacity Factor Router for inference-time load balancing
pub struct CapacityFactorRouter {
    config: CapacityConfig,
    queue_depths: Vec<AtomicUsize>,
}

impl CapacityFactorRouter {
    /// Create new router
    #[must_use]
    pub fn new(config: CapacityConfig) -> Self {
        let queue_depths = (0..config.num_experts)
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            config,
            queue_depths,
        }
    }

    /// Route to best expert, falling back if at capacity
    ///
    /// # Errors
    ///
    /// Returns `MoeError` if score count doesn't match expert count.
    /// Returns `ExpertCapacityExceeded` if all top experts are at capacity.
    pub fn route(&self, scores: &[f32]) -> Result<usize> {
        if scores.len() != self.config.num_experts {
            return Err(RealizarError::MoeError(format!(
                "Expected {} scores, got {}",
                self.config.num_experts,
                scores.len()
            )));
        }

        let top2 = Self::top_k_indices(scores, 2);
        let primary = top2[0];

        if self.queue_depths[primary].load(Ordering::Relaxed) < self.config.capacity {
            Ok(primary)
        } else if top2.len() > 1 {
            Ok(top2[1])
        } else {
            Err(RealizarError::ExpertCapacityExceeded {
                expert_id: primary,
                queue_depth: self.queue_depths[primary].load(Ordering::Relaxed),
                capacity: self.config.capacity,
            })
        }
    }

    /// Record expert usage
    pub fn record_start(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_add(1, Ordering::Relaxed);
    }

    /// Record expert completion
    pub fn record_end(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_sub(1, Ordering::Relaxed);
    }

    /// Get queue depth for expert
    #[must_use]
    pub fn queue_depth(&self, expert_id: usize) -> usize {
        self.queue_depths[expert_id].load(Ordering::Relaxed)
    }

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

// ============================================================================
// Power of Two Choices Router (Mitzenmacher 2001)
// ============================================================================

/// Configuration for Power of Two Choices routing
#[derive(Debug, Clone)]
pub struct PowerOfTwoConfig {
    /// Number of experts available
    pub num_experts: usize,
    /// Maximum queue depth per expert
    pub capacity: usize,
}

/// Power of Two Choices Router per Mitzenmacher (2001)
///
/// Instead of always routing to the highest-scoring expert, this router
/// picks the top 2 experts by score and routes to the *least loaded* one.
/// This dramatically improves load balancing compared to simple top-k routing.
///
/// ## Algorithm
///
/// 1. Select top-2 experts by score
/// 2. Compare their current queue depths
/// 3. Route to the one with lower load (breaking ties by score)
///
/// ## Citation
///
/// Mitzenmacher, M. (2001). "The Power of Two Choices in Randomized Load Balancing."
/// IEEE Transactions on Parallel and Distributed Systems.
pub struct PowerOfTwoChoicesRouter {
    config: PowerOfTwoConfig,
    queue_depths: Vec<AtomicUsize>,
}

impl PowerOfTwoChoicesRouter {
    /// Create a new Power of Two Choices router
    #[must_use]
    pub fn new(config: PowerOfTwoConfig) -> Self {
        let queue_depths = (0..config.num_experts)
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            config,
            queue_depths,
        }
    }

    /// Route request using Power of Two Choices algorithm
    ///
    /// # Errors
    ///
    /// Returns error if score count doesn't match expert count or all top experts at capacity.
    pub fn route(&self, scores: &[f32]) -> Result<usize> {
        if scores.len() != self.config.num_experts {
            return Err(RealizarError::MoeError(format!(
                "Expected {} scores, got {}",
                self.config.num_experts,
                scores.len()
            )));
        }

        // Get top 2 experts by score
        let top2 = Self::top_k_indices(scores, 2);

        // Check both for capacity and pick least loaded
        let mut best_choice = None;
        let mut best_load = usize::MAX;

        for &expert_id in &top2 {
            let load = self.queue_depths[expert_id].load(Ordering::Relaxed);
            if load < self.config.capacity && load < best_load {
                best_load = load;
                best_choice = Some(expert_id);
            }
        }

        best_choice.ok_or_else(|| RealizarError::ExpertCapacityExceeded {
            expert_id: top2[0],
            queue_depth: self.queue_depths[top2[0]].load(Ordering::Relaxed),
            capacity: self.config.capacity,
        })
    }

    /// Record that an expert started processing a request
    pub fn record_start(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_add(1, Ordering::Relaxed);
    }

    /// Record that an expert finished processing a request
    pub fn record_end(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current queue depth for an expert
    #[must_use]
    pub fn queue_depth(&self, expert_id: usize) -> usize {
        self.queue_depths[expert_id].load(Ordering::Relaxed)
    }

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

// ============================================================================
// Circuit Breaker (Nygard 2018)
// ============================================================================

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests flow through
    Closed,
    /// Failure threshold exceeded - requests blocked
    Open,
    /// Testing if service recovered - limited requests allowed
    HalfOpen,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening
    pub failure_threshold: usize,
    /// Number of successes needed to close from half-open
    pub success_threshold: usize,
    /// Time in milliseconds before transitioning from open to half-open
    pub timeout_ms: u64,
}

/// Circuit Breaker per Nygard (2018) "Release It!"
///
/// Prevents cascading failures by isolating failing components.
///
/// ## State Machine
///
/// ```text
/// CLOSED --[failures >= threshold]--> OPEN
///    ^                                  |
///    |                                  v
///    +--[successes >= threshold]-- HALF_OPEN <--[timeout]--+
/// ```
///
/// ## Citation
///
/// Nygard, M. (2018). "Release It! Design and Deploy Production-Ready Software."
/// Pragmatic Bookshelf, 2nd Edition.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    /// Protected mutable state
    state: Mutex<CircuitBreakerState>,
}

struct CircuitBreakerState {
    current: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(CircuitBreakerState {
                current: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
            }),
        }
    }

    /// Get current circuit state
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);
        state.current
    }

    /// Check if request should be allowed
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);

        match state.current {
            CircuitState::Open => false,
            CircuitState::Closed | CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn record_success(&self) {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);

        match state.current {
            CircuitState::Closed => {
                state.failure_count = 0; // Reset on success
            },
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= self.config.success_threshold {
                    state.current = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                }
            },
            CircuitState::Open => {}, // Shouldn't happen, but ignore
        }
    }

    /// Record a failed request
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn record_failure(&self) {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");

        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failure_count >= self.config.failure_threshold {
            state.current = CircuitState::Open;
            state.success_count = 0;
        }
    }

    fn maybe_transition_to_half_open(&self, state: &mut CircuitBreakerState) {
        if state.current == CircuitState::Open {
            if let Some(last_failure) = state.last_failure_time {
                let timeout = Duration::from_millis(self.config.timeout_ms);
                if last_failure.elapsed() >= timeout {
                    state.current = CircuitState::HalfOpen;
                    state.success_count = 0;
                }
            }
        }
    }
}

// ============================================================================
// Heijunka Controller (Toyota Production System)
// ============================================================================

/// Configuration for Heijunka (load leveling) controller
#[derive(Debug, Clone)]
pub struct HeijunkaConfig {
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    /// Maximum allowed concurrency
    pub max_concurrency: usize,
}

/// Load shedding decision
#[derive(Debug, Clone)]
pub struct LoadSheddingDecision {
    /// Whether to shed load (reject requests)
    pub shed_load: bool,
    /// Recommended concurrency level
    pub recommended_concurrency: usize,
}

/// Heijunka Controller for load leveling via Little's Law
///
/// Little's Law: L = λW
/// - L = average number of items in system (concurrency)
/// - λ = arrival rate (requests per second)
/// - W = average wait time (latency)
///
/// Rearranging: `optimal_concurrency = arrival_rate × (latency_ms / 1000)`
///
/// ## Toyota Production System
///
/// Heijunka (平準化) means "leveling" - smoothing production to avoid overburden.
/// In ML inference, this means maintaining steady throughput without latency spikes.
pub struct HeijunkaController {
    config: HeijunkaConfig,
}

impl HeijunkaController {
    /// Create a new Heijunka controller
    #[must_use]
    pub fn new(config: HeijunkaConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal concurrency using Little's Law
    ///
    /// # Arguments
    ///
    /// * `arrival_rate` - Requests per second
    /// * `latency_ms` - Average latency in milliseconds
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn optimal_concurrency(&self, arrival_rate: f64, latency_ms: f64) -> usize {
        // Little's Law: L = λW
        let optimal = (arrival_rate * latency_ms / 1000.0).ceil() as usize;
        optimal.clamp(1, self.config.max_concurrency)
    }

    /// Determine if load should be shed based on current state
    ///
    /// # Arguments
    ///
    /// * `current_latency_ms` - Current observed latency
    /// * `current_concurrency` - Current number of concurrent requests
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn should_shed_load(
        &self,
        current_latency_ms: f64,
        current_concurrency: usize,
    ) -> LoadSheddingDecision {
        let should_shed = current_latency_ms > self.config.target_latency_ms
            && current_concurrency >= self.config.max_concurrency;

        // Calculate recommended concurrency to meet target latency
        // If latency is 2x target, we need to halve concurrency
        let ratio = self.config.target_latency_ms / current_latency_ms;
        let concurrency_f64: f64 = current_concurrency as f64;
        let recommended = (concurrency_f64 * ratio).ceil() as usize;

        LoadSheddingDecision {
            shed_load: should_shed,
            recommended_concurrency: recommended.clamp(1, self.config.max_concurrency),
        }
    }

    /// Get the target latency
    #[must_use]
    pub fn target_latency_ms(&self) -> f64 {
        self.config.target_latency_ms
    }
}

/// Andon trigger types per Toyota Production System (Jidoka)
#[derive(Debug, Clone, PartialEq)]
pub enum AndonTrigger {
    /// Model checksum mismatch - corrupted model
    ModelChecksumMismatch {
        /// ID of the corrupted model
        model_id: String,
    },
    /// Latency P99 exceeded threshold
    LatencyExceeded {
        /// Observed P99 latency in milliseconds
        p99_ms: f64,
        /// Threshold that was exceeded
        threshold_ms: f64,
    },
    /// Error rate above threshold
    ErrorRateThreshold {
        /// Observed error rate (0.0 - 1.0)
        rate: f64,
        /// Threshold that was exceeded
        threshold: f64,
    },
    /// Expert load imbalance detected
    ExpertImbalance {
        /// Ratio of max/min expert utilization
        imbalance_ratio: f64,
    },
}

/// Response action for Andon triggers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AndonResponse {
    /// Automatically rollback to previous known-good state
    Rollback,
    /// Notify operators but continue serving
    Notify,
    /// Quarantine the failing expert (stop routing to it)
    Quarantine,
}

impl AndonTrigger {
    /// Determine appropriate response for this trigger
    #[must_use]
    pub fn response(&self) -> AndonResponse {
        match self {
            Self::ModelChecksumMismatch { .. } => AndonResponse::Rollback,
            Self::ErrorRateThreshold { rate, threshold } => {
                if *rate > threshold * 2.0 {
                    AndonResponse::Quarantine
                } else {
                    AndonResponse::Notify
                }
            },
            Self::LatencyExceeded { .. } | Self::ExpertImbalance { .. } => AndonResponse::Notify,
        }
    }

    /// Check if this trigger is critical (requires immediate action)
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(
            self.response(),
            AndonResponse::Rollback | AndonResponse::Quarantine
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Power of Two Choices Router Tests (Mitzenmacher 2001)
    // ========================================================================

    #[test]
    fn test_power_of_two_choices_selects_least_loaded() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 4,
            capacity: 100,
        });

        // Load expert 1 heavily
        for _ in 0..50 {
            router.record_start(1);
        }

        // With scores favoring experts 1 and 2, should pick least loaded (2)
        let scores = vec![0.1, 0.9, 0.8, 0.1];
        let choice = router.route(&scores).expect("test");

        // Should pick expert 2 (second best) since expert 1 is heavily loaded
        assert_eq!(choice, 2);
    }

    #[test]
    fn test_power_of_two_choices_equal_load_picks_best_score() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 4,
            capacity: 100,
        });

        // No load on any expert
        let scores = vec![0.1, 0.9, 0.8, 0.1];
        let choice = router.route(&scores).expect("test");

        // Should pick expert 1 (best score) since all equally loaded
        assert_eq!(choice, 1);
    }

    #[test]
    fn test_power_of_two_choices_respects_capacity() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 2,
            capacity: 5,
        });

        // Fill both experts to capacity
        for _ in 0..5 {
            router.record_start(0);
            router.record_start(1);
        }

        let scores = vec![0.9, 0.8];
        let result = router.route(&scores);

        // Should error - both at capacity
        assert!(result.is_err());
    }

    // ========================================================================
    // Circuit Breaker Tests (Nygard 2018)
    // ========================================================================

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 1000,
        });
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_ms: 1000,
        });

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_blocks_when_open() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout_ms: 100_000, // Long timeout
        });

        cb.record_failure();
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_half_open_after_timeout() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout_ms: 1, // Very short timeout
        });

        cb.record_failure();
        std::thread::sleep(std::time::Duration::from_millis(5));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.allow_request()); // Should allow probe request
    }

    #[test]
    fn test_circuit_breaker_closes_on_success_in_half_open() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_ms: 1,
        });

        cb.record_failure();
        std::thread::sleep(std::time::Duration::from_millis(5));

        // In half-open state
        cb.record_success();
        cb.record_success();

        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // ========================================================================
    // Heijunka Controller Tests (Toyota Production System)
    // ========================================================================

    #[test]
    fn test_heijunka_calculates_optimal_concurrency() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 100,
        });

        // Little's Law: L = λW
        // If arrival_rate = 10 req/s and latency = 100ms = 0.1s
        // Optimal concurrency = 10 * 0.1 = 1
        let concurrency = controller.optimal_concurrency(10.0, 100.0);
        assert_eq!(concurrency, 1);
    }

    #[test]
    fn test_heijunka_caps_at_max_concurrency() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 10,
        });

        // High arrival rate would want 100 concurrent, but capped at 10
        let concurrency = controller.optimal_concurrency(1000.0, 100.0);
        assert_eq!(concurrency, 10);
    }

    #[test]
    fn test_heijunka_load_leveling_decision() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 50,
        });

        // Current load exceeds target
        let decision = controller.should_shed_load(150.0, 50);
        assert!(decision.shed_load);

        // Current load under target
        let decision = controller.should_shed_load(50.0, 10);
        assert!(!decision.shed_load);
    }

    // ========================================================================
    // Original Capacity Factor Router Tests
    // ========================================================================

    #[test]
    fn test_route_to_best_expert() {
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 10,
            num_experts: 4,
        });
        let scores = vec![0.1, 0.5, 0.3, 0.1];
        assert_eq!(router.route(&scores).expect("test"), 1);
    }

    #[test]
    fn test_fallback_when_primary_full() {
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 1,
            num_experts: 4,
        });
        router.record_start(1); // Fill expert 1
        let scores = vec![0.1, 0.5, 0.3, 0.1];
        assert_eq!(router.route(&scores).expect("test"), 2); // Falls back to #2
    }

    #[test]
    fn test_queue_depth_tracking() {
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 10,
            num_experts: 2,
        });
        assert_eq!(router.queue_depth(0), 0);
        router.record_start(0);
        assert_eq!(router.queue_depth(0), 1);
        router.record_end(0);
        assert_eq!(router.queue_depth(0), 0);
    }

    #[test]
    fn test_wrong_score_count_error() {
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 10,
            num_experts: 4,
        });
        let scores = vec![0.5, 0.5]; // Wrong count
        assert!(router.route(&scores).is_err());
    }

    #[test]
    fn test_andon_checksum_triggers_rollback() {
        let trigger = AndonTrigger::ModelChecksumMismatch {
            model_id: "model-1".to_string(),
        };
        assert_eq!(trigger.response(), AndonResponse::Rollback);
        assert!(trigger.is_critical());
    }

    #[test]
    fn test_andon_latency_triggers_notify() {
        let trigger = AndonTrigger::LatencyExceeded {
            p99_ms: 150.0,
            threshold_ms: 100.0,
        };
        assert_eq!(trigger.response(), AndonResponse::Notify);
        assert!(!trigger.is_critical());
    }

    #[test]
    fn test_andon_high_error_rate_quarantines() {
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.25,
            threshold: 0.1,
        };
        assert_eq!(trigger.response(), AndonResponse::Quarantine);
        assert!(trigger.is_critical());
    }

    #[test]
    fn test_andon_moderate_error_rate_notifies() {
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.15,
            threshold: 0.1,
        };
        assert_eq!(trigger.response(), AndonResponse::Notify);
    }

    // ========================================================================
    // Additional Coverage Tests - Edge Cases and Error Paths
    // ========================================================================

    #[test]
    fn test_capacity_factor_router_single_expert_at_capacity() {
        // Test the edge case where there's only one expert and it's at capacity
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 1,
            num_experts: 1,
        });
        router.record_start(0); // Fill the only expert

        let scores = vec![0.9];
        let result = router.route(&scores);

        // Should return ExpertCapacityExceeded error
        assert!(result.is_err());
        match result {
            Err(RealizarError::ExpertCapacityExceeded {
                expert_id,
                queue_depth,
                capacity,
            }) => {
                assert_eq!(expert_id, 0);
                assert_eq!(queue_depth, 1);
                assert_eq!(capacity, 1);
            }
            _ => panic!("Expected ExpertCapacityExceeded error"),
        }
    }

    #[test]
    fn test_capacity_factor_router_both_top_experts_at_capacity() {
        // Test fallback when both top-2 experts are at capacity
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 1,
            num_experts: 4,
        });

        // Fill top 2 experts (index 1 and 2 based on scores)
        router.record_start(1);
        router.record_start(2);

        let scores = vec![0.1, 0.9, 0.8, 0.2];
        let result = router.route(&scores);

        // Primary (1) is full, fallback (2) is also full, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_power_of_two_choices_wrong_score_count() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 4,
            capacity: 10,
        });

        // Pass wrong number of scores
        let scores = vec![0.5, 0.5]; // Only 2, expecting 4
        let result = router.route(&scores);

        assert!(result.is_err());
        match result {
            Err(RealizarError::MoeError(msg)) => {
                assert!(msg.contains("Expected 4 scores, got 2"));
            }
            _ => panic!("Expected MoeError"),
        }
    }

    #[test]
    fn test_power_of_two_choices_queue_depth_tracking() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 3,
            capacity: 10,
        });

        assert_eq!(router.queue_depth(0), 0);
        assert_eq!(router.queue_depth(1), 0);

        router.record_start(1);
        router.record_start(1);
        assert_eq!(router.queue_depth(1), 2);

        router.record_end(1);
        assert_eq!(router.queue_depth(1), 1);

        router.record_end(1);
        assert_eq!(router.queue_depth(1), 0);
    }

    #[test]
    fn test_circuit_breaker_success_resets_failure_count_in_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_ms: 1000,
        });

        // Record some failures but not enough to open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        // Success should reset failure count
        cb.record_success();

        // Now we need 3 more failures to open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_success_while_open_no_effect() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_ms: 100_000, // Long timeout so stays open
        });

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Recording success while open should have no effect
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request()); // Still blocked
    }

    #[test]
    fn test_circuit_breaker_partial_success_in_half_open() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 3, // Need 3 successes
            timeout_ms: 1,
        });

        cb.record_failure();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // One success, still half-open
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Two successes, still half-open
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Three successes, now closed
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_heijunka_target_latency_getter() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 150.0,
            max_concurrency: 50,
        });

        assert!((controller.target_latency_ms() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_heijunka_minimum_concurrency_floor() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 100,
        });

        // Very low arrival rate should still give at least 1
        let concurrency = controller.optimal_concurrency(0.001, 1.0);
        assert_eq!(concurrency, 1);

        // Zero arrival rate (edge case)
        let concurrency = controller.optimal_concurrency(0.0, 100.0);
        assert_eq!(concurrency, 1);
    }

    #[test]
    fn test_heijunka_recommended_concurrency_bounds() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 10,
        });

        // Very high latency relative to target should recommend low concurrency
        let decision = controller.should_shed_load(1000.0, 10);
        assert!(decision.shed_load);
        assert!(decision.recommended_concurrency >= 1);
        assert!(decision.recommended_concurrency <= 10);

        // Very low latency should recommend higher concurrency (capped at current)
        let decision = controller.should_shed_load(10.0, 5);
        assert!(!decision.shed_load);
        assert!(decision.recommended_concurrency >= 1);
        assert!(decision.recommended_concurrency <= 10);
    }

    #[test]
    fn test_heijunka_no_shed_when_under_max_concurrency() {
        let controller = HeijunkaController::new(HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 50,
        });

        // High latency but under max concurrency - should not shed
        let decision = controller.should_shed_load(200.0, 30);
        assert!(!decision.shed_load);
    }

    #[test]
    fn test_andon_expert_imbalance_triggers_notify() {
        let trigger = AndonTrigger::ExpertImbalance {
            imbalance_ratio: 3.5,
        };
        assert_eq!(trigger.response(), AndonResponse::Notify);
        assert!(!trigger.is_critical());
    }

    #[test]
    fn test_andon_error_rate_exactly_at_threshold() {
        // Rate equal to threshold (not exceeding 2x) should notify
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.1,
            threshold: 0.1,
        };
        assert_eq!(trigger.response(), AndonResponse::Notify);
        assert!(!trigger.is_critical());
    }

    #[test]
    fn test_andon_error_rate_exactly_2x_threshold() {
        // Rate exactly 2x threshold should still notify (not strictly greater)
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.2,
            threshold: 0.1,
        };
        assert_eq!(trigger.response(), AndonResponse::Notify);
        assert!(!trigger.is_critical());

        // Rate slightly above 2x should quarantine
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.21,
            threshold: 0.1,
        };
        assert_eq!(trigger.response(), AndonResponse::Quarantine);
        assert!(trigger.is_critical());
    }

    #[test]
    fn test_capacity_factor_top_k_with_nan_scores() {
        // Test handling of NaN values in scores (via partial_cmp fallback)
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 10,
            num_experts: 3,
        });

        let scores = vec![f32::NAN, 0.5, 0.3];
        // This should not panic - NaN comparison falls back to Equal
        let result = router.route(&scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_power_of_two_choices_with_nan_scores() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 3,
            capacity: 10,
        });

        let scores = vec![f32::NAN, 0.5, 0.3];
        // Should not panic
        let result = router.route(&scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_capacity_factor_all_equal_scores() {
        let router = CapacityFactorRouter::new(CapacityConfig {
            capacity: 10,
            num_experts: 4,
        });

        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let result = router.route(&scores);
        assert!(result.is_ok());
        // When all scores are equal, should return a valid index
        assert!(result.unwrap() < 4);
    }

    #[test]
    fn test_power_of_two_single_expert() {
        let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
            num_experts: 1,
            capacity: 5,
        });

        let scores = vec![0.9];
        let result = router.route(&scores);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        // Fill to capacity
        for _ in 0..5 {
            router.record_start(0);
        }

        let result = router.route(&scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_circuit_breaker_failure_in_half_open_reopens() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 3,
            timeout_ms: 1,
        });

        // Get to half-open state
        cb.record_failure();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Failure in half-open should re-open the circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_capacity_config_debug() {
        let config = CapacityConfig {
            capacity: 10,
            num_experts: 4,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("capacity: 10"));
        assert!(debug_str.contains("num_experts: 4"));
    }

    #[test]
    fn test_power_of_two_config_debug() {
        let config = PowerOfTwoConfig {
            num_experts: 8,
            capacity: 20,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("num_experts: 8"));
        assert!(debug_str.contains("capacity: 20"));
    }

    #[test]
    fn test_circuit_breaker_config_debug() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 1000,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("failure_threshold: 5"));
        assert!(debug_str.contains("success_threshold: 3"));
        assert!(debug_str.contains("timeout_ms: 1000"));
    }

    #[test]
    fn test_heijunka_config_debug() {
        let config = HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 50,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("target_latency_ms: 100.0"));
        assert!(debug_str.contains("max_concurrency: 50"));
    }

    #[test]
    fn test_circuit_state_debug_and_eq() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::Open, CircuitState::Open);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open);

        let debug_str = format!("{:?}", CircuitState::HalfOpen);
        assert_eq!(debug_str, "HalfOpen");
    }

    #[test]
    fn test_load_shedding_decision_debug() {
        let decision = LoadSheddingDecision {
            shed_load: true,
            recommended_concurrency: 5,
        };
        let debug_str = format!("{:?}", decision);
        assert!(debug_str.contains("shed_load: true"));
        assert!(debug_str.contains("recommended_concurrency: 5"));
    }

    #[test]
    fn test_andon_trigger_debug_and_clone() {
        let trigger = AndonTrigger::ModelChecksumMismatch {
            model_id: "test-model".to_string(),
        };
        let cloned = trigger.clone();
        assert_eq!(trigger, cloned);

        let debug_str = format!("{:?}", trigger);
        assert!(debug_str.contains("ModelChecksumMismatch"));
        assert!(debug_str.contains("test-model"));
    }

    #[test]
    fn test_andon_response_debug_and_eq() {
        assert_eq!(AndonResponse::Rollback, AndonResponse::Rollback);
        assert_eq!(AndonResponse::Notify, AndonResponse::Notify);
        assert_eq!(AndonResponse::Quarantine, AndonResponse::Quarantine);
        assert_ne!(AndonResponse::Rollback, AndonResponse::Notify);

        let debug_str = format!("{:?}", AndonResponse::Quarantine);
        assert_eq!(debug_str, "Quarantine");
    }

    #[test]
    fn test_capacity_config_clone() {
        let config = CapacityConfig {
            capacity: 10,
            num_experts: 4,
        };
        let cloned = config.clone();
        assert_eq!(config.capacity, cloned.capacity);
        assert_eq!(config.num_experts, cloned.num_experts);
    }

    #[test]
    fn test_power_of_two_config_clone() {
        let config = PowerOfTwoConfig {
            num_experts: 8,
            capacity: 20,
        };
        let cloned = config.clone();
        assert_eq!(config.num_experts, cloned.num_experts);
        assert_eq!(config.capacity, cloned.capacity);
    }

    #[test]
    fn test_circuit_breaker_config_clone() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 1000,
        };
        let cloned = config.clone();
        assert_eq!(config.failure_threshold, cloned.failure_threshold);
        assert_eq!(config.success_threshold, cloned.success_threshold);
        assert_eq!(config.timeout_ms, cloned.timeout_ms);
    }

    #[test]
    fn test_heijunka_config_clone() {
        let config = HeijunkaConfig {
            target_latency_ms: 100.0,
            max_concurrency: 50,
        };
        let cloned = config.clone();
        assert!((config.target_latency_ms - cloned.target_latency_ms).abs() < f64::EPSILON);
        assert_eq!(config.max_concurrency, cloned.max_concurrency);
    }

    #[test]
    fn test_load_shedding_decision_clone() {
        let decision = LoadSheddingDecision {
            shed_load: true,
            recommended_concurrency: 5,
        };
        let cloned = decision.clone();
        assert_eq!(decision.shed_load, cloned.shed_load);
        assert_eq!(
            decision.recommended_concurrency,
            cloned.recommended_concurrency
        );
    }

    #[test]
    fn test_andon_trigger_latency_clone_and_eq() {
        let trigger = AndonTrigger::LatencyExceeded {
            p99_ms: 150.0,
            threshold_ms: 100.0,
        };
        let cloned = trigger.clone();
        assert_eq!(trigger, cloned);
    }

    #[test]
    fn test_andon_trigger_error_rate_clone_and_eq() {
        let trigger = AndonTrigger::ErrorRateThreshold {
            rate: 0.15,
            threshold: 0.1,
        };
        let cloned = trigger.clone();
        assert_eq!(trigger, cloned);
    }

    #[test]
    fn test_andon_trigger_imbalance_clone_and_eq() {
        let trigger = AndonTrigger::ExpertImbalance {
            imbalance_ratio: 2.5,
        };
        let cloned = trigger.clone();
        assert_eq!(trigger, cloned);
    }

    #[test]
    fn test_circuit_state_copy() {
        let state = CircuitState::Closed;
        let copied = state;
        assert_eq!(state, copied);
    }

    #[test]
    fn test_andon_response_clone_all_variants() {
        // Test clone behavior for all AndonResponse variants
        let rollback = AndonResponse::Rollback;
        let rollback_cloned = rollback.clone();
        assert_eq!(rollback, rollback_cloned);

        let notify = AndonResponse::Notify;
        let notify_cloned = notify.clone();
        assert_eq!(notify, notify_cloned);

        let quarantine = AndonResponse::Quarantine;
        let quarantine_cloned = quarantine.clone();
        assert_eq!(quarantine, quarantine_cloned);
    }
}
