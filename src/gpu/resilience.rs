//! GPU Resilience Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - Retry, circuit breaker, and bulkhead patterns.
//!
//! ## Contents
//! - `RetryPolicy`, `RetryConfig` - Exponential backoff retry (IMP-076)
//! - `CircuitBreaker`, `CircuitConfig` - Circuit breaker pattern (IMP-077)
//! - `BulkheadManager`, `BulkheadConfig` - Resource isolation (IMP-078)

use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// M31: Retry Logic & Circuit Breakers (IMP-076, IMP-077, IMP-078)
// ============================================================================

/// Error category for retry decisions (IMP-076)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient error that may succeed on retry
    Transient,
    /// Permanent error that will not succeed on retry
    Permanent,
}

/// Retry decision (IMP-076)
#[derive(Debug, Clone)]
pub enum RetryDecision {
    /// Retry with specified delay
    Retry {
        /// Delay before retry
        delay: Duration,
    },
    /// Abort with reason
    Abort {
        /// Reason for abort
        reason: String,
    },
}

/// Retry configuration (IMP-076)
#[derive(Debug, Clone)]
pub struct RetryConfig {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter_factor: f64,
}

impl RetryConfig {
    /// Create new retry config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
        }
    }

    /// Set max retries
    #[must_use]
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set base delay
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set max delay
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set jitter factor (0.0 to 1.0)
    #[must_use]
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry policy (IMP-076)
pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    /// Create new retry policy
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Get max retries
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    /// Decide whether to retry
    #[must_use]
    pub fn should_retry(&self, attempt: u32, category: ErrorCategory) -> RetryDecision {
        if category == ErrorCategory::Permanent {
            return RetryDecision::Abort {
                reason: "Permanent error".to_string(),
            };
        }

        if attempt > self.config.max_retries {
            return RetryDecision::Abort {
                reason: format!("Max retries ({}) exceeded", self.config.max_retries),
            };
        }

        RetryDecision::Retry {
            delay: self.calculate_delay(attempt),
        }
    }

    /// Calculate delay with exponential backoff
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        // Exponential backoff: base * 2^attempt
        let exp_delay_ms = self.config.base_delay.as_millis() as u64 * (1u64 << attempt.min(20));
        let delay_ms = exp_delay_ms.min(self.config.max_delay.as_millis() as u64);
        Duration::from_millis(delay_ms)
    }
}

/// Circuit breaker state (IMP-077)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests allowed
    Closed,
    /// Circuit is open, requests rejected
    Open,
    /// Circuit is half-open, probe requests allowed
    HalfOpen,
}

/// Circuit breaker configuration (IMP-077)
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
}

impl CircuitConfig {
    /// Create new circuit config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set failure threshold to open circuit
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set success threshold to close circuit from half-open
    #[must_use]
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set timeout before transitioning to half-open
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker (IMP-077)
pub struct CircuitBreaker {
    config: CircuitConfig,
    state: std::sync::Mutex<CircuitState>,
    failure_count: std::sync::atomic::AtomicU32,
    success_count: std::sync::atomic::AtomicU32,
    last_failure: std::sync::Mutex<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[must_use]
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: std::sync::Mutex::new(CircuitState::Closed),
            failure_count: std::sync::atomic::AtomicU32::new(0),
            success_count: std::sync::atomic::AtomicU32::new(0),
            last_failure: std::sync::Mutex::new(None),
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitState {
        *self.state.lock().expect("mutex poisoned")
    }

    /// Check if request should be allowed
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.lock().expect("mutex poisoned");
        match *state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_failure = self.last_failure.lock().expect("mutex poisoned");
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.timeout {
                        *state = CircuitState::HalfOpen;
                        self.success_count
                            .store(0, std::sync::atomic::Ordering::SeqCst);
                        return true;
                    }
                }
                false
            },
        }
    }

    /// Record a failure
    pub fn record_failure(&self) {
        let count = self
            .failure_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        *self.last_failure.lock().expect("mutex poisoned") = Some(std::time::Instant::now());

        let mut state = self.state.lock().expect("mutex poisoned");
        match *state {
            CircuitState::Closed => {
                if count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                }
            },
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
            },
            CircuitState::Open => {},
        }
    }

    /// Record a success
    pub fn record_success(&self) {
        self.failure_count
            .store(0, std::sync::atomic::Ordering::SeqCst);
        let count = self
            .success_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        let mut state = self.state.lock().expect("mutex poisoned");
        if *state == CircuitState::HalfOpen && count >= self.config.success_threshold {
            *state = CircuitState::Closed;
        }
    }
}

/// Request type for bulkhead (IMP-078)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestType {
    /// Standard inference request
    Inference,
    /// Embedding generation request
    Embedding,
    /// Batch processing request
    Batch,
}

/// Bulkhead permit
#[derive(Debug)]
pub struct BulkheadPermit {
    request_type: RequestType,
}

/// Bulkhead configuration (IMP-078)
pub struct BulkheadConfig {
    pools: HashMap<String, usize>,
}

impl BulkheadConfig {
    /// Create new bulkhead config
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Add a pool with specified size
    #[must_use]
    pub fn with_pool(mut self, name: &str, size: usize) -> Self {
        self.pools.insert(name.to_string(), size);
        self
    }
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Bulkhead stats
#[derive(Debug, Clone)]
pub struct BulkheadStats {
    /// Number of pools
    pub pool_count: usize,
    /// Total capacity across all pools
    pub total_capacity: usize,
}

/// Bulkhead manager (IMP-078)
pub struct BulkheadManager {
    inference_available: std::sync::atomic::AtomicUsize,
    inference_capacity: usize,
    embedding_available: std::sync::atomic::AtomicUsize,
    embedding_capacity: usize,
    batch_available: std::sync::atomic::AtomicUsize,
    batch_capacity: usize,
}

impl BulkheadManager {
    /// Create new bulkhead manager
    #[must_use]
    pub fn new(config: &BulkheadConfig) -> Self {
        let inference_cap = *config.pools.get("inference").unwrap_or(&10);
        let embedding_cap = *config.pools.get("embedding").unwrap_or(&5);
        let batch_cap = *config.pools.get("batch").unwrap_or(&2);

        Self {
            inference_available: std::sync::atomic::AtomicUsize::new(inference_cap),
            inference_capacity: inference_cap,
            embedding_available: std::sync::atomic::AtomicUsize::new(embedding_cap),
            embedding_capacity: embedding_cap,
            batch_available: std::sync::atomic::AtomicUsize::new(batch_cap),
            batch_capacity: batch_cap,
        }
    }

    /// Get available slots for request type
    #[must_use]
    pub fn available(&self, request_type: RequestType) -> usize {
        match request_type {
            RequestType::Inference => self
                .inference_available
                .load(std::sync::atomic::Ordering::SeqCst),
            RequestType::Embedding => self
                .embedding_available
                .load(std::sync::atomic::Ordering::SeqCst),
            RequestType::Batch => self
                .batch_available
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Acquire a permit
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        let available = match request_type {
            RequestType::Inference => &self.inference_available,
            RequestType::Embedding => &self.embedding_available,
            RequestType::Batch => &self.batch_available,
        };

        let current = available.load(std::sync::atomic::Ordering::SeqCst);
        if current == 0 {
            return Err("Pool exhausted");
        }
        available.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(BulkheadPermit { request_type })
    }

    /// Try to acquire a permit (non-blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn try_acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        self.acquire(request_type)
    }

    /// Release a permit
    pub fn release(&self, permit: &BulkheadPermit) {
        let available = match permit.request_type {
            RequestType::Inference => &self.inference_available,
            RequestType::Embedding => &self.embedding_available,
            RequestType::Batch => &self.batch_available,
        };
        available.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get bulkhead stats
    #[must_use]
    pub fn stats(&self) -> BulkheadStats {
        BulkheadStats {
            pool_count: 3,
            total_capacity: self.inference_capacity + self.embedding_capacity + self.batch_capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ErrorCategory Tests ====================

    #[test]
    fn test_error_category_transient() {
        let cat = ErrorCategory::Transient;
        assert_eq!(cat, ErrorCategory::Transient);
        assert_ne!(cat, ErrorCategory::Permanent);
    }

    #[test]
    fn test_error_category_permanent() {
        let cat = ErrorCategory::Permanent;
        assert_eq!(cat, ErrorCategory::Permanent);
    }

    #[test]
    fn test_error_category_clone_copy() {
        let cat = ErrorCategory::Transient;
        let cloned = cat.clone();
        let copied: ErrorCategory = cat;
        assert_eq!(cat, cloned);
        assert_eq!(cat, copied);
    }

    // ==================== RetryDecision Tests ====================

    #[test]
    fn test_retry_decision_retry() {
        let decision = RetryDecision::Retry {
            delay: Duration::from_millis(100),
        };
        if let RetryDecision::Retry { delay } = decision {
            assert_eq!(delay.as_millis(), 100);
        } else {
            panic!("Expected Retry variant");
        }
    }

    #[test]
    fn test_retry_decision_abort() {
        let decision = RetryDecision::Abort {
            reason: "test".to_string(),
        };
        if let RetryDecision::Abort { reason } = decision {
            assert_eq!(reason, "test");
        } else {
            panic!("Expected Abort variant");
        }
    }

    #[test]
    fn test_retry_decision_clone() {
        let decision = RetryDecision::Retry {
            delay: Duration::from_secs(1),
        };
        let cloned = decision.clone();
        if let RetryDecision::Retry { delay } = cloned {
            assert_eq!(delay.as_secs(), 1);
        } else {
            panic!("Expected Retry variant");
        }
    }

    // ==================== RetryConfig Tests ====================

    #[test]
    fn test_retry_config_new() {
        let config = RetryConfig::new();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay.as_millis(), 100);
        assert_eq!(config.max_delay.as_secs(), 30);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_retry_config_with_max_retries() {
        let config = RetryConfig::new().with_max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_retry_config_with_base_delay() {
        let config = RetryConfig::new().with_base_delay(Duration::from_millis(200));
        assert_eq!(config.base_delay.as_millis(), 200);
    }

    #[test]
    fn test_retry_config_with_max_delay() {
        let config = RetryConfig::new().with_max_delay(Duration::from_secs(60));
        assert_eq!(config.max_delay.as_secs(), 60);
    }

    #[test]
    fn test_retry_config_with_jitter_factor() {
        let config = RetryConfig::new().with_jitter_factor(0.2);
        assert!((config.jitter_factor - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_retry_config_jitter_clamped() {
        let config = RetryConfig::new().with_jitter_factor(2.0);
        assert!((config.jitter_factor - 1.0).abs() < 1e-6);

        let config2 = RetryConfig::new().with_jitter_factor(-0.5);
        assert!((config2.jitter_factor - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_retry_config_clone() {
        let config = RetryConfig::new().with_max_retries(10);
        let cloned = config.clone();
        assert_eq!(cloned.max_retries, 10);
    }

    // ==================== RetryPolicy Tests ====================

    #[test]
    fn test_retry_policy_new() {
        let config = RetryConfig::new();
        let policy = RetryPolicy::new(config);
        assert_eq!(policy.max_retries(), 3);
    }

    #[test]
    fn test_retry_policy_should_retry_transient() {
        let config = RetryConfig::new().with_max_retries(3);
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(1, ErrorCategory::Transient);
        assert!(matches!(decision, RetryDecision::Retry { .. }));
    }

    #[test]
    fn test_retry_policy_should_retry_permanent() {
        let config = RetryConfig::new();
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(1, ErrorCategory::Permanent);
        if let RetryDecision::Abort { reason } = decision {
            assert!(reason.contains("Permanent"));
        } else {
            panic!("Expected Abort for permanent error");
        }
    }

    #[test]
    fn test_retry_policy_max_retries_exceeded() {
        let config = RetryConfig::new().with_max_retries(2);
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(3, ErrorCategory::Transient);
        if let RetryDecision::Abort { reason } = decision {
            assert!(reason.contains("exceeded"));
        } else {
            panic!("Expected Abort when max retries exceeded");
        }
    }

    #[test]
    fn test_retry_policy_calculate_delay_exponential() {
        let config = RetryConfig::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(60));
        let policy = RetryPolicy::new(config);

        // attempt 0: 100 * 2^0 = 100ms
        assert_eq!(policy.calculate_delay(0).as_millis(), 100);
        // attempt 1: 100 * 2^1 = 200ms
        assert_eq!(policy.calculate_delay(1).as_millis(), 200);
        // attempt 2: 100 * 2^2 = 400ms
        assert_eq!(policy.calculate_delay(2).as_millis(), 400);
    }

    #[test]
    fn test_retry_policy_calculate_delay_capped() {
        let config = RetryConfig::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(500));
        let policy = RetryPolicy::new(config);

        // attempt 5: 100 * 2^5 = 3200ms, but capped at 500ms
        let delay = policy.calculate_delay(5);
        assert_eq!(delay.as_millis(), 500);
    }

    // ==================== CircuitState Tests ====================

    #[test]
    fn test_circuit_state_values() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::Open, CircuitState::Open);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open);
    }

    #[test]
    fn test_circuit_state_clone_copy() {
        let state = CircuitState::HalfOpen;
        let cloned = state.clone();
        let copied: CircuitState = state;
        assert_eq!(state, cloned);
        assert_eq!(state, copied);
    }

    // ==================== CircuitConfig Tests ====================

    #[test]
    fn test_circuit_config_new() {
        let config = CircuitConfig::new();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 2);
        assert_eq!(config.timeout.as_secs(), 30);
    }

    #[test]
    fn test_circuit_config_default() {
        let config = CircuitConfig::default();
        assert_eq!(config.failure_threshold, 5);
    }

    #[test]
    fn test_circuit_config_with_failure_threshold() {
        let config = CircuitConfig::new().with_failure_threshold(10);
        assert_eq!(config.failure_threshold, 10);
    }

    #[test]
    fn test_circuit_config_with_success_threshold() {
        let config = CircuitConfig::new().with_success_threshold(3);
        assert_eq!(config.success_threshold, 3);
    }

    #[test]
    fn test_circuit_config_with_timeout() {
        let config = CircuitConfig::new().with_timeout(Duration::from_secs(60));
        assert_eq!(config.timeout.as_secs(), 60);
    }

    #[test]
    fn test_circuit_config_clone() {
        let config = CircuitConfig::new().with_failure_threshold(8);
        let cloned = config.clone();
        assert_eq!(cloned.failure_threshold, 8);
    }

    // ==================== CircuitBreaker Tests ====================

    #[test]
    fn test_circuit_breaker_new() {
        let config = CircuitConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let config = CircuitConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert!(breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_record_failure_opens() {
        let config = CircuitConfig::new().with_failure_threshold(2);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_open_blocks_requests() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_timeout(Duration::from_secs(60));
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_record_success_resets_failures() {
        let config = CircuitConfig::new().with_failure_threshold(3);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        breaker.record_success(); // Should reset failure count
        breaker.record_failure();
        breaker.record_failure();
        // Still closed because success reset the count
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_success_threshold(2)
            .with_timeout(Duration::from_millis(1));
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout to transition to half-open
        std::thread::sleep(Duration::from_millis(5));
        assert!(breaker.allow_request()); // Should transition to HalfOpen
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes to close
        breaker.record_success();
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_timeout(Duration::from_millis(1));
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        breaker.allow_request(); // Transition to HalfOpen
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_failure(); // Should reopen
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    // ==================== RequestType Tests ====================

    #[test]
    fn test_request_type_values() {
        assert_eq!(RequestType::Inference, RequestType::Inference);
        assert_eq!(RequestType::Embedding, RequestType::Embedding);
        assert_eq!(RequestType::Batch, RequestType::Batch);
        assert_ne!(RequestType::Inference, RequestType::Batch);
    }

    #[test]
    fn test_request_type_clone_copy() {
        let rt = RequestType::Embedding;
        let cloned = rt.clone();
        let copied: RequestType = rt;
        assert_eq!(rt, cloned);
        assert_eq!(rt, copied);
    }

    // ==================== BulkheadConfig Tests ====================

    #[test]
    fn test_bulkhead_config_new() {
        let config = BulkheadConfig::new();
        assert!(config.pools.is_empty());
    }

    #[test]
    fn test_bulkhead_config_default() {
        let config = BulkheadConfig::default();
        assert!(config.pools.is_empty());
    }

    #[test]
    fn test_bulkhead_config_with_pool() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 10)
            .with_pool("embedding", 5);
        assert_eq!(config.pools.get("inference"), Some(&10));
        assert_eq!(config.pools.get("embedding"), Some(&5));
    }

    // ==================== BulkheadStats Tests ====================

    #[test]
    fn test_bulkhead_stats_fields() {
        let stats = BulkheadStats {
            pool_count: 3,
            total_capacity: 17,
        };
        assert_eq!(stats.pool_count, 3);
        assert_eq!(stats.total_capacity, 17);
    }

    #[test]
    fn test_bulkhead_stats_clone() {
        let stats = BulkheadStats {
            pool_count: 2,
            total_capacity: 10,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.pool_count, stats.pool_count);
    }

    // ==================== BulkheadPermit Tests ====================

    #[test]
    fn test_bulkhead_permit_debug() {
        let permit = BulkheadPermit {
            request_type: RequestType::Inference,
        };
        let debug_str = format!("{:?}", permit);
        assert!(debug_str.contains("Inference"));
    }

    // ==================== BulkheadManager Tests ====================

    #[test]
    fn test_bulkhead_manager_new_defaults() {
        let config = BulkheadConfig::new();
        let manager = BulkheadManager::new(&config);

        assert_eq!(manager.available(RequestType::Inference), 10);
        assert_eq!(manager.available(RequestType::Embedding), 5);
        assert_eq!(manager.available(RequestType::Batch), 2);
    }

    #[test]
    fn test_bulkhead_manager_new_custom() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 20)
            .with_pool("embedding", 8);
        let manager = BulkheadManager::new(&config);

        assert_eq!(manager.available(RequestType::Inference), 20);
        assert_eq!(manager.available(RequestType::Embedding), 8);
    }

    #[test]
    fn test_bulkhead_manager_acquire_release() {
        let config = BulkheadConfig::new().with_pool("inference", 2);
        let manager = BulkheadManager::new(&config);

        let permit1 = manager.acquire(RequestType::Inference).unwrap();
        assert_eq!(manager.available(RequestType::Inference), 1);

        let permit2 = manager.acquire(RequestType::Inference).unwrap();
        assert_eq!(manager.available(RequestType::Inference), 0);

        manager.release(&permit1);
        assert_eq!(manager.available(RequestType::Inference), 1);

        manager.release(&permit2);
        assert_eq!(manager.available(RequestType::Inference), 2);
    }

    #[test]
    fn test_bulkhead_manager_acquire_exhausted() {
        let config = BulkheadConfig::new().with_pool("batch", 1);
        let manager = BulkheadManager::new(&config);

        let _permit = manager.acquire(RequestType::Batch).unwrap();
        let result = manager.acquire(RequestType::Batch);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Pool exhausted");
    }

    #[test]
    fn test_bulkhead_manager_try_acquire() {
        let config = BulkheadConfig::new().with_pool("embedding", 1);
        let manager = BulkheadManager::new(&config);

        let permit = manager.try_acquire(RequestType::Embedding).unwrap();
        assert_eq!(manager.available(RequestType::Embedding), 0);

        let result = manager.try_acquire(RequestType::Embedding);
        assert!(result.is_err());

        manager.release(&permit);
        assert_eq!(manager.available(RequestType::Embedding), 1);
    }

    #[test]
    fn test_bulkhead_manager_stats() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 10)
            .with_pool("embedding", 5)
            .with_pool("batch", 2);
        let manager = BulkheadManager::new(&config);

        let stats = manager.stats();
        assert_eq!(stats.pool_count, 3);
        assert_eq!(stats.total_capacity, 17);
    }

    #[test]
    fn test_bulkhead_manager_isolation() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 2)
            .with_pool("embedding", 2);
        let manager = BulkheadManager::new(&config);

        // Exhaust inference pool
        let _p1 = manager.acquire(RequestType::Inference).unwrap();
        let _p2 = manager.acquire(RequestType::Inference).unwrap();

        // Embedding pool should still be available
        let result = manager.acquire(RequestType::Embedding);
        assert!(result.is_ok());
    }
}
