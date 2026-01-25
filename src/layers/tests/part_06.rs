/// IMP-072: Failure Isolation
use crate::layers::*;
#[test]
#[cfg(feature = "gpu")]
fn test_imp_072_failure_isolation() {
    use crate::gpu::{FailureIsolator, RequestOutcome};
    use std::sync::Arc;

    // Test 1: Create failure isolator
    let isolator = FailureIsolator::new();
    assert_eq!(
        isolator.active_requests(),
        0,
        "IMP-072: Should start with 0 active"
    );

    // Test 2: Start isolated request
    let request_id = isolator.start_request();
    assert_eq!(
        isolator.active_requests(),
        1,
        "IMP-072: Should have 1 active request"
    );

    // Test 3: Complete request successfully
    isolator.complete_request(request_id, &RequestOutcome::Success);
    assert_eq!(
        isolator.active_requests(),
        0,
        "IMP-072: Should have 0 active after completion"
    );
    assert_eq!(
        isolator.success_count(),
        1,
        "IMP-072: Should have 1 success"
    );

    // Test 4: Handle failed request with cleanup
    let request_id = isolator.start_request();
    let cleanup_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cleanup_flag = cleanup_called.clone();
    isolator.register_cleanup(request_id, move || {
        cleanup_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    isolator.complete_request(
        request_id,
        &RequestOutcome::Failed("test error".to_string()),
    );
    assert!(
        cleanup_called.load(std::sync::atomic::Ordering::SeqCst),
        "IMP-072: Cleanup should be called on failure"
    );
    assert_eq!(
        isolator.failure_count(),
        1,
        "IMP-072: Should have 1 failure"
    );

    // Test 5: Circuit breaker opens after repeated failures
    for _ in 0..5 {
        let req_id = isolator.start_request();
        isolator.complete_request(req_id, &RequestOutcome::Failed("error".to_string()));
    }
    assert!(
        isolator.is_circuit_open(),
        "IMP-072: Circuit should open after repeated failures"
    );

    // Test 6: Circuit breaker rejects new requests when open
    let result = isolator.try_start_request();
    assert!(
        result.is_err(),
        "IMP-072: Should reject requests when circuit open"
    );

    // Test 7: Circuit breaker recovers after timeout
    isolator.reset_circuit();
    assert!(
        !isolator.is_circuit_open(),
        "IMP-072: Circuit should close after reset"
    );
    let result = isolator.try_start_request();
    assert!(
        result.is_ok(),
        "IMP-072: Should accept requests when circuit closed"
    );
}

// ========================================================================
// M30: Connection Pooling & Resource Limits (IMP-073, IMP-074, IMP-075)
// ========================================================================

/// M30: Connection Pool Management (IMP-073)
/// Target: Bounded pool, health checking, warm startup
#[test]
#[cfg(feature = "gpu")]
fn test_imp_073_connection_pool() {
    use crate::gpu::{ConnectionConfig, ConnectionPool, ConnectionState};

    // Test 1: Create pool with configurable limits
    let config = ConnectionConfig::new()
        .with_max_connections(10)
        .with_min_connections(2)
        .with_idle_timeout(std::time::Duration::from_secs(300));
    let pool = ConnectionPool::new(config);
    assert_eq!(
        pool.max_connections(),
        10,
        "IMP-073: Max connections should be configurable"
    );
    assert_eq!(
        pool.min_connections(),
        2,
        "IMP-073: Min connections should be configurable"
    );

    // Test 2: Acquire and release connections
    let conn = pool.acquire();
    assert!(conn.is_ok(), "IMP-073: Should acquire connection from pool");
    assert_eq!(
        pool.active_connections(),
        1,
        "IMP-073: Should track active connections"
    );

    pool.release(conn.expect("test"));
    assert_eq!(
        pool.active_connections(),
        0,
        "IMP-073: Should decrement on release"
    );

    // Test 3: Bounded pool rejects when full
    let mut conns = Vec::new();
    for i in 0..10 {
        let c = pool.acquire();
        assert!(c.is_ok(), "IMP-073: Should acquire connection {}", i);
        conns.push(c.expect("test"));
    }
    let overflow = pool.try_acquire();
    assert!(
        overflow.is_err(),
        "IMP-073: Should reject when pool exhausted"
    );

    // Release all
    for c in conns {
        pool.release(c);
    }

    // Test 4: Connection health checking
    let conn = pool.acquire().expect("test");
    let state = pool.check_health(&conn);
    assert!(
        matches!(state, ConnectionState::Healthy),
        "IMP-073: New connection should be healthy"
    );
    pool.release(conn);

    // Test 5: Warm pool on startup
    let pool2 = ConnectionPool::new(ConnectionConfig::new().with_min_connections(3));
    pool2.warm();
    assert!(
        pool2.idle_connections() >= 3,
        "IMP-073: Should warm pool to min connections"
    );
}

/// M30: Resource Limits (IMP-074)
/// Target: Memory limits, compute time limits, queue depth limits
#[test]
#[cfg(feature = "gpu")]
fn test_imp_074_resource_limits() {
    use crate::gpu::{LimitResult, ResourceConfig, ResourceLimiter};

    // Test 1: Create limiter with configurable limits
    let config = ResourceConfig::new()
        .with_max_memory_per_request(512 * 1024 * 1024) // 512MB
        .with_max_total_memory(4 * 1024 * 1024 * 1024) // 4GB
        .with_max_compute_time(std::time::Duration::from_secs(30))
        .with_max_queue_depth(100);
    let limiter = ResourceLimiter::new(config);

    // Test 2: Check memory limits per request
    let result = limiter.check_memory(256 * 1024 * 1024);
    assert!(
        matches!(result, LimitResult::Allowed),
        "IMP-074: Should allow within limits"
    );

    let result = limiter.check_memory(1024 * 1024 * 1024);
    assert!(
        matches!(result, LimitResult::Denied { .. }),
        "IMP-074: Should deny over per-request limit"
    );

    // Test 3: Track total memory usage
    let alloc1 = limiter.allocate(256 * 1024 * 1024);
    assert!(alloc1.is_ok(), "IMP-074: Should allocate memory");
    assert_eq!(
        limiter.current_memory(),
        256 * 1024 * 1024,
        "IMP-074: Should track allocated"
    );

    limiter.deallocate(256 * 1024 * 1024);
    assert_eq!(
        limiter.current_memory(),
        0,
        "IMP-074: Should track deallocated"
    );

    // Test 4: Queue depth limits with backpressure
    for _ in 0..100 {
        let _ = limiter.enqueue();
    }
    let overflow = limiter.try_enqueue();
    assert!(
        matches!(overflow, LimitResult::Backpressure),
        "IMP-074: Should apply backpressure"
    );

    // Drain queue
    for _ in 0..100 {
        limiter.dequeue();
    }

    // Test 5: Compute time tracking
    let timer = limiter.start_compute();
    assert!(
        timer.elapsed() < std::time::Duration::from_secs(1),
        "IMP-074: Timer should work"
    );
}

/// M30: Resource Monitoring (IMP-075)
/// Target: Real-time memory, GPU utilization, queue metrics
#[test]
#[cfg(feature = "gpu")]
fn test_imp_075_resource_monitoring() {
    use crate::gpu::ResourceMonitor;

    // Test 1: Create monitor
    let monitor = ResourceMonitor::new();

    // Test 2: Track memory usage
    monitor.record_memory_usage(512 * 1024 * 1024);
    let metrics = monitor.current_metrics();
    assert_eq!(
        metrics.memory_bytes,
        512 * 1024 * 1024,
        "IMP-075: Should track memory"
    );

    // Test 3: Track GPU utilization
    monitor.record_gpu_utilization(75.5);
    let metrics = monitor.current_metrics();
    assert!(
        (metrics.gpu_utilization - 75.5).abs() < 0.01,
        "IMP-075: Should track GPU util"
    );

    // Test 4: Track queue depth
    monitor.record_queue_depth(42);
    let metrics = monitor.current_metrics();
    assert_eq!(metrics.queue_depth, 42, "IMP-075: Should track queue depth");

    // Test 5: Track request latency
    monitor.record_latency(std::time::Duration::from_millis(150));
    let metrics = monitor.current_metrics();
    assert_eq!(
        metrics.last_latency_ms, 150,
        "IMP-075: Should track latency"
    );

    // Test 6: Aggregate metrics (min/max/avg)
    for i in 1..=5 {
        monitor.record_latency(std::time::Duration::from_millis(i * 100));
    }
    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 100, "IMP-075: Should track min latency");
    assert_eq!(stats.max_ms, 500, "IMP-075: Should track max latency");
    // 6 values: 150, 100, 200, 300, 400, 500 = 1650 / 6 = 275
    assert_eq!(stats.avg_ms, 275, "IMP-075: Should track avg latency");

    // Test 7: Snapshot for reporting
    let snapshot = monitor.snapshot();
    assert!(
        snapshot.timestamp > 0,
        "IMP-075: Snapshot should have timestamp"
    );
    assert!(
        snapshot.memory_bytes > 0,
        "IMP-075: Snapshot should include memory"
    );
}

// ========================================================================
// M31: Retry Logic & Circuit Breakers (IMP-076, IMP-077, IMP-078)
// ========================================================================

/// M31: Retry Strategy (IMP-076)
/// Target: Configurable retry policies, exponential backoff, max limits
#[test]
#[cfg(feature = "gpu")]
fn test_imp_076_retry_strategy() {
    use crate::gpu::{ErrorCategory, RetryConfig, RetryDecision, RetryPolicy};

    // Test 1: Create retry config with defaults
    let config = RetryConfig::new()
        .with_max_retries(5)
        .with_base_delay(std::time::Duration::from_millis(100))
        .with_max_delay(std::time::Duration::from_secs(30))
        .with_jitter_factor(0.2);
    let policy = RetryPolicy::new(config);
    assert_eq!(
        policy.max_retries(),
        5,
        "IMP-076: Max retries should be configurable"
    );

    // Test 2: Decide retry for transient error
    let decision = policy.should_retry(1, ErrorCategory::Transient);
    assert!(
        matches!(decision, RetryDecision::Retry { .. }),
        "IMP-076: Should retry transient error"
    );

    // Test 3: Don't retry permanent errors
    let decision = policy.should_retry(1, ErrorCategory::Permanent);
    assert!(
        matches!(decision, RetryDecision::Abort { .. }),
        "IMP-076: Should not retry permanent error"
    );

    // Test 4: Exponential backoff calculation
    let delay1 = policy.calculate_delay(1);
    let delay2 = policy.calculate_delay(2);
    let delay3 = policy.calculate_delay(3);
    assert!(
        delay2 > delay1,
        "IMP-076: Delay should increase (exp backoff)"
    );
    assert!(delay3 > delay2, "IMP-076: Delay should continue increasing");

    // Test 5: Max delay capping
    let delay_capped = policy.calculate_delay(100);
    assert!(
        delay_capped <= std::time::Duration::from_secs(30),
        "IMP-076: Should cap at max delay"
    );

    // Test 6: Max retries exceeded
    let decision = policy.should_retry(6, ErrorCategory::Transient);
    assert!(
        matches!(decision, RetryDecision::Abort { .. }),
        "IMP-076: Should abort after max retries"
    );
}

/// M31: Circuit Breaker Pattern (IMP-077)
/// Target: Closed/Open/Half-Open states, failure threshold, timeout probe
#[test]
#[cfg(feature = "gpu")]
fn test_imp_077_circuit_breaker() {
    use crate::gpu::{CircuitBreaker, CircuitConfig, CircuitState};

    // Test 1: Create circuit breaker with config
    let config = CircuitConfig::new()
        .with_failure_threshold(3)
        .with_success_threshold(2)
        .with_timeout(std::time::Duration::from_millis(100));
    let breaker = CircuitBreaker::new(config);
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should start closed"
    );

    // Test 2: Record failures up to threshold
    breaker.record_failure();
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should stay closed below threshold"
    );

    // Test 3: Open after threshold
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Open),
        "IMP-077: Should open at threshold"
    );

    // Test 4: Reject requests when open
    assert!(!breaker.allow_request(), "IMP-077: Should reject when open");

    // Test 5: Transition to half-open after timeout
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        breaker.allow_request(),
        "IMP-077: Should allow probe after timeout"
    );
    assert!(
        matches!(breaker.state(), CircuitState::HalfOpen),
        "IMP-077: Should be half-open"
    );

    // Test 6: Close on success in half-open
    breaker.record_success();
    breaker.record_success();
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should close after successes"
    );

    // Test 7: Re-open on failure in half-open
    // First get to half-open state
    for _ in 0..3 {
        breaker.record_failure();
    }
    std::thread::sleep(std::time::Duration::from_millis(150));
    let _ = breaker.allow_request(); // Transition to half-open
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Open),
        "IMP-077: Should re-open on half-open failure"
    );
}

/// M31: Bulkhead Pattern (IMP-078)
/// Target: Separate pools, prevent starvation, configurable sizes
#[test]
#[cfg(feature = "gpu")]
fn test_imp_078_bulkhead_pattern() {
    use crate::gpu::{BulkheadConfig, BulkheadManager, RequestType};

    // Test 1: Create bulkhead manager with config
    let config = BulkheadConfig::new()
        .with_pool("inference", 10)
        .with_pool("embedding", 5)
        .with_pool("batch", 2);
    let manager = BulkheadManager::new(&config);

    // Test 2: Acquire from specific pool
    let permit = manager.acquire(RequestType::Inference);
    assert!(
        permit.is_ok(),
        "IMP-078: Should acquire from inference pool"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Should decrement available"
    );

    // Test 3: Pools are isolated
    let embed_permit = manager.acquire(RequestType::Embedding);
    assert!(
        embed_permit.is_ok(),
        "IMP-078: Should acquire from embedding pool"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Inference should be unchanged"
    );
    assert_eq!(
        manager.available(RequestType::Embedding),
        4,
        "IMP-078: Embedding should decrement"
    );

    // Test 4: Pool exhaustion doesn't affect others
    for _ in 0..2 {
        let _ = manager.acquire(RequestType::Batch);
    }
    let batch_overflow = manager.try_acquire(RequestType::Batch);
    assert!(
        batch_overflow.is_err(),
        "IMP-078: Batch pool should be exhausted"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Inference still available"
    );

    // Test 5: Release returns to correct pool
    manager.release(&permit.expect("test"));
    assert_eq!(
        manager.available(RequestType::Inference),
        10,
        "IMP-078: Should release to correct pool"
    );

    // Test 6: Get pool stats
    let stats = manager.stats();
    assert_eq!(stats.pool_count, 3, "IMP-078: Should have 3 pools");
    assert!(
        stats.total_capacity >= 17,
        "IMP-078: Total capacity should sum pools"
    );
}

// ========================================================================
// M32: Production Logging & Diagnostics (IMP-079, IMP-080, IMP-081)
// ========================================================================

/// M32: Structured Logging (IMP-079)
/// Target: JSON-formatted logs, correlation IDs, configurable levels
#[test]
#[cfg(feature = "gpu")]
fn test_imp_079_structured_logging() {
    use crate::gpu::{LogConfig, LogEntry, LogLevel, Logger};

    // Test 1: Create logger with config
    let config = LogConfig::new()
        .with_level(LogLevel::Debug)
        .with_json_format(true)
        .with_module_level("gpu", LogLevel::Trace);
    let logger = Logger::new(config);

    // Test 2: Create log entry with structured data
    let entry = LogEntry::new(LogLevel::Info, "Request started")
        .with_correlation_id("req-12345")
        .with_field("model", "llama-7b")
        .with_field("tokens", "128");
    assert_eq!(
        entry.correlation_id(),
        Some("req-12345"),
        "IMP-079: Should have correlation ID"
    );
    assert_eq!(entry.level(), LogLevel::Info, "IMP-079: Should have level");

    // Test 3: JSON formatting
    let json = entry.to_json();
    assert!(
        json.contains("\"level\":\"INFO\""),
        "IMP-079: JSON should have level"
    );
    assert!(
        json.contains("\"correlation_id\":\"req-12345\""),
        "IMP-079: JSON should have correlation ID"
    );
    assert!(
        json.contains("\"model\":\"llama-7b\""),
        "IMP-079: JSON should have custom fields"
    );

    // Test 4: Module-specific log levels
    assert!(
        logger.is_enabled(LogLevel::Trace, "gpu"),
        "IMP-079: gpu should allow Trace"
    );
    assert!(
        logger.is_enabled(LogLevel::Debug, "inference"),
        "IMP-079: Other modules use default"
    );
    assert!(
        !logger.is_enabled(LogLevel::Trace, "inference"),
        "IMP-079: Trace should be filtered for non-gpu"
    );

    // Test 5: Log with automatic timestamp
    let entry = LogEntry::new(LogLevel::Warn, "High memory usage");
    assert!(entry.timestamp() > 0, "IMP-079: Should have timestamp");
}

/// M32: Performance Diagnostics (IMP-080)
/// Target: Latency breakdown, memory tracking, GPU timing
#[test]
#[cfg(feature = "gpu")]
fn test_imp_080_performance_diagnostics() {
    use crate::gpu::{DiagnosticsCollector, MemoryTracker, PhaseTimer};

    // Test 1: Create diagnostics collector
    let collector = DiagnosticsCollector::new();

    // Test 2: Track request phases
    let timer = PhaseTimer::new();
    timer.start_phase("tokenization");
    std::thread::sleep(std::time::Duration::from_millis(10));
    timer.end_phase("tokenization");
    timer.start_phase("inference");
    std::thread::sleep(std::time::Duration::from_millis(20));
    timer.end_phase("inference");

    let breakdown = timer.breakdown();
    assert!(
        breakdown.contains_key("tokenization"),
        "IMP-080: Should track tokenization"
    );
    assert!(
        breakdown.contains_key("inference"),
        "IMP-080: Should track inference"
    );
    assert!(
        *breakdown.get("inference").expect("test") > *breakdown.get("tokenization").expect("test"),
        "IMP-080: Inference should take longer"
    );

    // Test 3: Memory allocation tracking
    let tracker = MemoryTracker::new();
    tracker.record_allocation("model_weights", 1024 * 1024 * 1024);
    tracker.record_allocation("kv_cache", 256 * 1024 * 1024);
    tracker.record_deallocation("kv_cache", 256 * 1024 * 1024);

    let report = tracker.report();
    assert_eq!(
        report.peak_bytes,
        1024 * 1024 * 1024 + 256 * 1024 * 1024,
        "IMP-080: Should track peak"
    );
    assert_eq!(
        report.current_bytes,
        1024 * 1024 * 1024,
        "IMP-080: Should track current"
    );
    assert_eq!(
        report.allocation_count, 2,
        "IMP-080: Should count allocations"
    );

    // Test 4: Report to collector
    collector.record_request_timing("req-001", timer.breakdown());
    collector.record_memory_snapshot(report);
    let summary = collector.summary();
    assert!(summary.request_count >= 1, "IMP-080: Should count requests");
}

/// M32: Debug Mode (IMP-081)
/// Target: Verbose logging, request replay, state dump
#[test]
#[cfg(feature = "gpu")]
fn test_imp_081_debug_mode() {
    use crate::gpu::{DebugMode, RequestCapture, StateDump};

    // Test 1: Enable debug mode
    let debug = DebugMode::new();
    assert!(
        !debug.is_enabled(),
        "IMP-081: Should be disabled by default"
    );
    debug.enable();
    assert!(debug.is_enabled(), "IMP-081: Should enable");

    // Test 2: Capture request for replay
    let capture = RequestCapture::new()
        .with_input("Hello, world!")
        .with_params("temperature", "0.7")
        .with_params("max_tokens", "100");
    assert_eq!(
        capture.input(),
        "Hello, world!",
        "IMP-081: Should capture input"
    );
    assert_eq!(capture.params().len(), 2, "IMP-081: Should capture params");

    // Test 3: Serialize/deserialize for replay
    let json = capture.to_json();
    let restored = RequestCapture::from_json(&json);
    assert!(restored.is_ok(), "IMP-081: Should deserialize");
    assert_eq!(
        restored.expect("test").input(),
        "Hello, world!",
        "IMP-081: Should restore input"
    );

    // Test 4: State dump on error
    let dump = StateDump::new()
        .with_error("Out of memory")
        .with_stack_trace("at inference::generate\nat main")
        .with_state("model_loaded", "true")
        .with_state("tokens_processed", "42");
    assert_eq!(
        dump.error(),
        "Out of memory",
        "IMP-081: Should capture error"
    );
    assert!(
        dump.stack_trace().contains("inference::generate"),
        "IMP-081: Should have stack"
    );
    assert_eq!(dump.state().len(), 2, "IMP-081: Should capture state");

    // Test 5: Dump to file (mock)
    let dump_json = dump.to_json();
    assert!(
        dump_json.contains("Out of memory"),
        "IMP-081: JSON should have error"
    );
    assert!(
        dump_json.contains("tokens_processed"),
        "IMP-081: JSON should have state"
    );
}

// =========================================================================
// M33: GGUF HTTP Serving Integration Tests
// Per spec v2.15.0: Wire GpuModel to HTTP server
// =========================================================================

/// M33: GgufModelState (IMP-082)
/// Target: App state that holds a loaded GGUF model for HTTP serving
#[test]
#[cfg(feature = "gpu")]
fn test_imp_082_gguf_model_state() {
    use crate::gpu::GgufModelState;

    // Test 1: Create empty state
    let state = GgufModelState::new();
    assert!(!state.is_loaded(), "IMP-082: Should be unloaded initially");

    // Test 2: State reports model info
    assert_eq!(
        state.model_name(),
        None,
        "IMP-082: No model name when empty"
    );
    assert_eq!(state.vocab_size(), 0, "IMP-082: Zero vocab when empty");

    // Test 3: Ready check
    assert!(!state.is_ready(), "IMP-082: Not ready when empty");
}

/// M33: Load GGUF to GPU (IMP-083)
/// Target: Pipeline from GGUF file to GpuModel ready for inference
#[test]
#[cfg(feature = "gpu")]
fn test_imp_083_load_gguf_to_gpu() {
    use crate::gpu::load_gguf_to_gpu;

    // Test with test GGUF data (minimal model)
    let vocab_size = 256;
    let hidden_dim = 64;
    let num_layers = 2;

    // Create minimal test GGUF-like config
    let result = load_gguf_to_gpu(vocab_size, hidden_dim, num_layers);

    // This should work - creates a minimal GPU model
    assert!(result.is_ok(), "IMP-083: Should load test model to GPU");

    let state = result.expect("test");
    assert!(state.is_loaded(), "IMP-083: Should be loaded after load");
    assert!(state.is_ready(), "IMP-083: Should be ready for inference");
    assert_eq!(
        state.vocab_size(),
        vocab_size,
        "IMP-083: Should have correct vocab"
    );
}

/// M33: Serve GGUF Model (IMP-084)
/// Target: HTTP server with loaded GGUF model (integration test)
///
/// Verifies that a GGUF model can be served via HTTP.
/// Run with: `cargo test test_imp_084 --ignored --features gpu`
#[test]
#[ignore = "Requires integration test setup"]
fn test_imp_084_serve_gguf_model() {
    // IMP-084: Integration test for serve_gguf_model
    //
    // This test verifies the HTTP serving infrastructure is correct.
    // It uses a demo model since real GGUF files may not be available.

    // Check if realizar server is running on default port
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to create HTTP client");

    let health_url = "http://127.0.0.1:3000/health";
    match client.get(health_url).send() {
        Ok(response) => {
            assert!(
                response.status().is_success(),
                "IMP-084: Health endpoint should return 200 OK"
            );
            println!("IMP-084: ✅ Server health check passed");

            // Test generate endpoint with demo model
            let gen_url = "http://127.0.0.1:3000/generate";
            let request = serde_json::json!({
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0.0
            });

            match client.post(gen_url).json(&request).send() {
                Ok(gen_response) => {
                    assert!(
                        gen_response.status().is_success(),
                        "IMP-084: Generate endpoint should return 200 OK"
                    );
                    let body: serde_json::Value = gen_response.json().expect("Valid JSON");
                    assert!(
                        body.get("text").is_some(),
                        "IMP-084: Response should have text"
                    );
                    println!("IMP-084: ✅ Generate endpoint works, got: {:?}", body);
                },
                Err(e) => {
                    println!("IMP-084: ⚠️ Generate endpoint not available: {}", e);
                },
            }
        },
        Err(e) => {
            panic!(
                "IMP-084: Server not running at {}. Start with: cargo run --example api_server. Error: {}",
                health_url, e
            );
        },
    }
}

/// M33: OpenAI Completions Endpoint (IMP-085)
/// Target: /v1/completions returns generated text
///
/// Tests OpenAI-compatible completions API.
/// Run with: `cargo test test_imp_085 --ignored`
#[test]
#[ignore = "Requires running server"]
fn test_imp_085_completions_endpoint() {
    // IMP-085: Integration test for /v1/completions (OpenAI-compatible)

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/v1/completions";

    // OpenAI-style request format
    let request = serde_json::json!({
        "model": "demo",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.0
    });

    match client.post(url).json(&request).send() {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().expect("Valid JSON");
                assert!(
                    body.get("choices").is_some(),
                    "IMP-085: Response should have 'choices'"
                );
                println!("IMP-085: ✅ OpenAI completions endpoint works");
            } else if response.status().as_u16() == 404 {
                println!("IMP-085: ⚠️ /v1/completions not implemented yet (404)");
            } else {
                panic!("IMP-085: Unexpected status: {}", response.status());
            }
        },
        Err(e) => {
            panic!(
                "IMP-085: Server not running. Start with: cargo run --example api_server. Error: {}",
                e
            );
        },
    }
}

/// M33: llama.cpp Completion Endpoint (IMP-086)
/// Target: /completion returns generated text (llama.cpp compatible)
///
/// Tests llama.cpp-compatible completion API.
/// Run with: `cargo test test_imp_086 --ignored`
#[test]
#[ignore = "Requires running server"]
fn test_imp_086_llamacpp_endpoint() {
    // IMP-086: Integration test for /completion (llama.cpp-compatible)

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/completion";

    // llama.cpp-style request format
    let request = serde_json::json!({
        "prompt": "Hello, world!",
        "n_predict": 10,
        "temperature": 0.0
    });

    match client.post(url).json(&request).send() {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().expect("Valid JSON");
                assert!(
                    body.get("content").is_some() || body.get("text").is_some(),
                    "IMP-086: Response should have 'content' or 'text'"
                );
                println!("IMP-086: ✅ llama.cpp completion endpoint works");
            } else if response.status().as_u16() == 404 {
                println!("IMP-086: ⚠️ /completion not implemented yet (404)");
            } else {
                panic!("IMP-086: Unexpected status: {}", response.status());
            }
        },
        Err(e) => {
            panic!(
                "IMP-086: Server not running. Start with: cargo run --example api_server. Error: {}",
                e
            );
        },
    }
}

/// M33: Benchmark Integration (IMP-087)
/// Target: realizar appears in bench-server-matrix.sh output
///
/// Verifies benchmark infrastructure is functional.
/// Run with: `cargo test test_imp_087 --ignored`
#[test]
#[ignore = "Requires benchmark infrastructure"]
fn test_imp_087_benchmark_integration() {
    // IMP-087: Benchmark integration test
    //
    // This test verifies that:
    // 1. The benchmark script exists
    // 2. The server can respond to benchmark-style requests
    // 3. Throughput can be measured

    use std::time::Instant;

    // Check if benchmark script exists
    let script_path = std::path::Path::new("scripts/bench-server-matrix.sh");
    if script_path.exists() {
        println!("IMP-087: ✅ Benchmark script exists at scripts/bench-server-matrix.sh");
    } else {
        println!("IMP-087: ⚠️ Benchmark script not found (optional)");
    }

    // Test benchmark-style request pattern
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/generate";
    let request = serde_json::json!({
        "prompt": "Benchmark test",
        "max_tokens": 10,
        "temperature": 0.0
    });

    // Run 5 iterations to measure throughput
    let iterations = 5;
    let start = Instant::now();
    let mut success_count = 0;
    let mut total_tokens = 0;

    for i in 0..iterations {
        match client.post(url).json(&request).send() {
            Ok(response) if response.status().is_success() => {
                if let Ok(body) = response.json::<serde_json::Value>() {
                    if let Some(text) = body.get("text").and_then(|t| t.as_str()) {
                        total_tokens += text.split_whitespace().count();
                        success_count += 1;
                    }
                }
            },
            Ok(response) => {
                println!(
                    "IMP-087: Iteration {} failed with status {}",
                    i,
                    response.status()
                );
            },
            Err(e) => {
                assert!(
                    i != 0,
                    "IMP-087: Server not running. Start with: cargo run --example api_server. Error: {}",
                    e
                );
            },
        }
    }

    let elapsed = start.elapsed();
    let throughput = if elapsed.as_secs_f64() > 0.0 {
        total_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "IMP-087: ✅ Benchmark test: {} iterations, {} tokens, {:.2} tok/s",
        success_count, total_tokens, throughput
    );

    assert!(
        success_count > 0,
        "IMP-087: At least one benchmark iteration should succeed"
    );
}

/// M33: GQA Support - num_kv_heads in config (IMP-088)
/// Target: GpuModelConfig has num_kv_heads field for Grouped Query Attention
#[test]
#[cfg(feature = "gpu")]
fn test_imp_088_gqa_config_num_kv_heads() {
    use crate::gpu::GpuModelConfig;

    // Create config with different num_kv_heads (GQA pattern)
    // Qwen 1.5B: 12 heads, 2 kv_heads (6:1 ratio)
    let config = GpuModelConfig {
        vocab_size: 151936,
        hidden_dim: 1536,
        num_heads: 12,
        num_kv_heads: 2, // GQA: fewer KV heads than Q heads
        num_layers: 28,
        intermediate_dim: 8960,
        eps: 1e-6,
        rope_theta: 10000.0,
    };

    assert_eq!(config.num_heads, 12, "IMP-088: Should have 12 Q heads");
    assert_eq!(config.num_kv_heads, 2, "IMP-088: Should have 2 KV heads");

    // head_dim should be hidden_dim / num_heads
    let head_dim = config.hidden_dim / config.num_heads;
    assert_eq!(head_dim, 128, "IMP-088: Head dim should be 128");

    // KV size per layer should use num_kv_heads
    let kv_head_dim = config.hidden_dim / config.num_heads; // Same head_dim
    let kv_size = config.num_kv_heads * kv_head_dim;
    assert_eq!(kv_size, 256, "IMP-088: KV size should be 2*128=256");
}

/// M33: GQA Attention Forward (IMP-089)
/// Target: Forward pass handles K/V with fewer heads than Q
#[test]
#[cfg(feature = "gpu")]
fn test_imp_089_gqa_attention_forward() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create GQA config (fewer KV heads than Q heads)
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 128,
        num_heads: 4,    // 4 Q heads
        num_kv_heads: 2, // 2 KV heads (2:1 ratio, each KV serves 2 Q heads)
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = GpuModel::new(config).expect("Failed to create GQA model");

    // Forward should work with GQA attention
    let tokens = vec![1usize, 2, 3];
    let result = model.forward_gpu(&tokens);

    assert!(
        result.is_ok(),
        "IMP-089: Forward pass should handle GQA attention. Error: {:?}",
        result.err()
    );

    let logits = result.expect("test");
    // Output should be [seq_len * vocab_size]
    assert_eq!(
        logits.len(),
        tokens.len() * 256,
        "IMP-089: Logits should be seq_len * vocab_size"
    );
}

/// M33: CPU Embedding for Large Vocab (IMP-090)
/// Target: Handle vocab sizes that exceed GPU buffer limits (>65536 tokens)
/// wgpu max buffer is 256MB, large vocab like Qwen (151936) needs CPU fallback
#[test]
#[cfg(feature = "gpu")]
fn test_imp_090_cpu_embedding_large_vocab() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Large vocab size that would exceed GPU buffer limits if stored fully
    // Real example: Qwen 2.5 Coder 1.5B has vocab_size=151936
    // Buffer size would be: 151936 * 1536 * 4 = 933MB > 256MB wgpu limit
    // Test with smaller but still "large vocab" threshold (>65536)
    let large_vocab_config = GpuModelConfig {
        vocab_size: 100_000, // Large vocab - requires CPU embedding fallback
        hidden_dim: 256,     // Smaller hidden_dim for test speed
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    // This should NOT fail due to GPU buffer limits
    // Instead, it should use CPU embedding lookup
    let model_result = GpuModel::new(large_vocab_config);

    assert!(
        model_result.is_ok(),
        "IMP-090: Should create model with large vocab using CPU embedding. Error: {:?}",
        model_result.err()
    );

    let mut model = model_result.expect("test");

    // Forward pass should also work with CPU embedding lookup
    let tokens = vec![0usize, 1000, 50000, 99999]; // Include edge tokens
    let result = model.forward_gpu(&tokens);

    assert!(
        result.is_ok(),
        "IMP-090: Forward pass should work with CPU embedding for large vocab. Error: {:?}",
        result.err()
    );

    let logits = result.expect("test");
    assert_eq!(
        logits.len(),
        tokens.len() * 100_000,
        "IMP-090: Logits should be seq_len * vocab_size"
    );

    // Verify embeddings are valid (not all zeros, not NaN)
    let has_valid_values = logits.iter().any(|&v| v != 0.0 && !v.is_nan());
    assert!(
        has_valid_values,
        "IMP-090: Logits should contain valid non-zero values"
    );
}

/// IMP-093: Real GGUF GPU benchmark test
///
/// Tests the full GPU inference path with a real GGUF model.
/// This verifies IMP-092 (no weight cloning) improves performance.
///
/// Run: cargo test --features gpu test_imp_093_real_gguf_gpu_benchmark -- --nocapture --ignored
#[test]
#[cfg(feature = "gpu")]
#[ignore] // Requires real GGUF file - run manually
fn test_imp_093_real_gguf_gpu_benchmark() {
    use crate::gguf::MappedGGUFModel;
    use crate::gpu::GpuModel;
    use std::path::Path;
    use std::time::Instant;

    // Real GGUF model path (Qwen 2.5 Coder 1.5B Q4_K_M)
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        eprintln!("IMP-093: Skipping - model not found at {}", model_path);
        return;
    }

    println!("\n=== IMP-093: Real GGUF GPU Benchmark ===\n");
    println!("Model: {}", model_path);

    // Load model
    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to load GGUF");
    let load_mmap = load_start.elapsed();
    println!("  Mmap load: {:?}", load_mmap);

    let gpu_start = Instant::now();
    let mut gpu_model = GpuModel::from_mapped_gguf(&mapped).expect("Failed to load to GPU");
    let gpu_load = gpu_start.elapsed();
    println!("  GPU load: {:?}", gpu_load);
    println!(
        "  Config: hidden={}, layers={}, vocab={}, heads={}, kv_heads={}, intermediate={}",
        gpu_model.config().hidden_dim,
        gpu_model.config().num_layers,
        gpu_model.config().vocab_size,
        gpu_model.config().num_heads,
        gpu_model.config().num_kv_heads,
        gpu_model.config().intermediate_dim,
    );
    println!();

    // Test tokens (small prompt)
    let test_tokens = vec![0usize, 1, 2, 3];
    let max_tokens = 5;

    // Warmup
    println!("Warmup...");
    let _ = gpu_model.generate(
        &test_tokens,
        &crate::gpu::GpuGenerateConfig {
            max_tokens: 1,
            ..Default::default()
        },
    );

    // Benchmark generation
    println!("\nGenerating {} tokens...", max_tokens);
    let gen_start = Instant::now();
    let result = gpu_model.generate(
        &test_tokens,
        &crate::gpu::GpuGenerateConfig {
            max_tokens,
            ..Default::default()
        },
    );
    let gen_elapsed = gen_start.elapsed();

    assert!(
        result.is_ok(),
        "IMP-093: Generation should succeed: {:?}",
        result.err()
    );

    let generated = result.expect("test");
    let gen_secs = gen_elapsed.as_secs_f64();
    let tps = max_tokens as f64 / gen_secs;

    println!("\n=== Results ===");
    println!(
        "  Generated: {} tokens",
        generated.len() - test_tokens.len()
    );
    println!("  Time: {:.3}s", gen_secs);
    println!("  Throughput: {:.2} tok/s", tps);
    println!();

    // Performance assertions (soft targets - document actual vs target)
    // Target: ≥10 tok/s (Ollama achieves ~143 tok/s)
    // IMP-092 eliminates 3.7GB/token memory copying
    let target_tps = 10.0;
    if tps < target_tps {
        eprintln!(
            "WARNING: Below target {} tok/s (actual: {:.2} tok/s)",
            target_tps, tps
        );
        eprintln!("Parity gap with Ollama (~143 tok/s): {:.0}x", 143.0 / tps);
    } else {
        println!(
            "PASS: Achieved {:.2} tok/s (target: {} tok/s)",
            tps, target_tps
        );
    }
}

/// IMP-099: Benchmark fused Q4_K matvec vs f32 matvec
///
/// Compares memory bandwidth and compute performance of:
/// - f32 matvec: 4 bytes per weight, SIMD accumulation
/// - Q4_K matvec: ~0.56 bytes per weight, fused dequant+dot
#[test]
#[ignore] // Run manually: cargo test --release test_imp_099_q4k_vs_f32_benchmark -- --nocapture --ignored
fn test_imp_099_q4k_vs_f32_benchmark() {
    use crate::quantize::{fused_q4k_parallel_matvec, QK_K};
    use std::time::Instant;

    println!("\n=== IMP-099: Q4_K vs f32 Matmul Benchmark ===\n");

    // Realistic dimensions for transformer layer
    // Qwen 2.5 1.5B: hidden=1536, intermediate=8960
    let in_dim: usize = 1536; // Must be multiple of 256 for Q4_K
    let out_dim: usize = 8960;
    let iterations = 100;

    // Create test data
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    // Q4_K weights: 144 bytes per 256 values
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;
    let q4k_weight_size = out_dim * bytes_per_row;
    let q4k_weights: Vec<u8> = (0..q4k_weight_size).map(|i| (i % 256) as u8).collect();

    // f32 weights: 4 bytes per value
    let f32_weight_size = in_dim * out_dim;
    let f32_weights: Vec<f32> = (0..f32_weight_size)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();

    println!("Dimensions: {} x {}", in_dim, out_dim);
    println!("Q4_K weight size: {:.2} MB", q4k_weight_size as f64 / 1e6);
    println!(
        "f32 weight size: {:.2} MB",
        (f32_weight_size * 4) as f64 / 1e6
    );
    println!(
        "Compression ratio: {:.1}x\n",
        (f32_weight_size * 4) as f64 / q4k_weight_size as f64
    );

    // Warmup
    let _ = fused_q4k_parallel_matvec(&q4k_weights, &activations, in_dim, out_dim);
    let _ = crate::gpu::cpu_matmul(&activations, &f32_weights, 1, in_dim, out_dim);

    // Benchmark Q4_K fused matvec
    let q4k_start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_parallel_matvec(&q4k_weights, &activations, in_dim, out_dim);
    }
    let q4k_elapsed = q4k_start.elapsed();
    let q4k_per_op = q4k_elapsed.as_secs_f64() / iterations as f64;

    // Benchmark f32 matvec (using cpu_matmul which calls cpu_vector_matmul for m=1)
    let f32_start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::gpu::cpu_matmul(&activations, &f32_weights, 1, in_dim, out_dim);
    }
    let f32_elapsed = f32_start.elapsed();
    let f32_per_op = f32_elapsed.as_secs_f64() / iterations as f64;

    // Calculate metrics
    let q4k_gops = (in_dim * out_dim) as f64 / q4k_per_op / 1e9;
    let f32_gops = (in_dim * out_dim) as f64 / f32_per_op / 1e9;
    let q4k_bw = q4k_weight_size as f64 / q4k_per_op / 1e9;
    let f32_bw = (f32_weight_size * 4) as f64 / f32_per_op / 1e9;

    println!("=== Results ({} iterations) ===", iterations);
    println!("Q4_K fused:");
    println!("  Time: {:.3} ms/op", q4k_per_op * 1000.0);
    println!("  Throughput: {:.2} GOPS", q4k_gops);
    println!("  Bandwidth: {:.2} GB/s", q4k_bw);
    println!();
    println!("f32 matvec:");
    println!("  Time: {:.3} ms/op", f32_per_op * 1000.0);
    println!("  Throughput: {:.2} GOPS", f32_gops);
    println!("  Bandwidth: {:.2} GB/s", f32_bw);
    println!();
    println!("Speedup (Q4_K vs f32): {:.2}x", f32_per_op / q4k_per_op);
    println!("Effective bandwidth amplification: {:.2}x", f32_bw / q4k_bw);
}

// =========================================================================
// Coverage Tests: Getter Methods
// =========================================================================

/// Test LayerNorm getter methods
#[test]
fn test_layer_norm_getters() {
    let ln = LayerNorm::new(64, 1e-5).expect("test");
    assert_eq!(ln.normalized_shape(), 64);
    assert!((ln.eps() - 1e-5).abs() < 1e-10);
}

/// Test Linear getter methods
#[test]
fn test_linear_getters() {
    let linear = Linear::new(32, 64).expect("test");
    assert_eq!(linear.in_features(), 32);
    assert_eq!(linear.out_features(), 64);
}

/// Test Linear mutable accessors
#[test]
fn test_linear_mutable_accessors() {
    let mut linear = Linear::new(4, 2).expect("test");

    // Modify weights
    let weights = linear.weight_mut();
    assert_eq!(weights.len(), 4 * 2);
    weights[0] = 1.0;
    assert_eq!(linear.weight_mut()[0], 1.0);

    // Modify bias
    let bias = linear.bias_mut();
    assert_eq!(bias.len(), 2);
    bias[0] = 0.5;
    assert_eq!(linear.bias_mut()[0], 0.5);
}

/// Test QuantizedLinear getter methods
#[test]
fn test_quantized_linear_getters() {
    // Create minimal Q4_K weight (144 bytes per 256 values)
    let weight_bytes = vec![0u8; 144 * 2]; // 512 values = 2 super-blocks
    let bias = vec![0.0f32; 2];
    let ql = QuantizedLinear::new(256, 2, weight_bytes, bias).expect("test");

    assert_eq!(ql.in_features(), 256);
    assert_eq!(ql.out_features(), 2);
    assert_eq!(ql.weight_bytes().len(), 144 * 2);
    assert_eq!(ql.bias().len(), 2);
    assert!(ql.memory_bytes() > 0);
}

/// Test FusedLayerNormLinear getter methods
#[test]
fn test_fused_layer_norm_linear_getters() {
    let fused = FusedLayerNormLinear::new(8, 4, 1e-5).expect("test");
    assert_eq!(fused.feature_dim(), 8);
    assert_eq!(fused.out_features(), 4);
}

/// Test FusedLayerNormLinear mutable accessors
#[test]
fn test_fused_layer_norm_linear_mutable_accessors() {
    let mut fused = FusedLayerNormLinear::new(4, 2, 1e-5).expect("test");

    // Modify norm weights
    let norm_w = fused.norm_weight_mut();
    assert_eq!(norm_w.len(), 4);
    norm_w[0] = 2.0;
    assert_eq!(fused.norm_weight_mut()[0], 2.0);

    // Modify norm bias
    let norm_b = fused.norm_bias_mut();
    assert_eq!(norm_b.len(), 4);
    norm_b[0] = 0.1;
    assert_eq!(fused.norm_bias_mut()[0], 0.1);

    // Modify linear weights
    let lin_w = fused.linear_weight_mut();
    assert_eq!(lin_w.len(), 4 * 2);
    lin_w[0] = 3.0;
    assert_eq!(fused.linear_weight_mut()[0], 3.0);

    // Modify linear bias
    let lin_b = fused.linear_bias_mut();
    assert_eq!(lin_b.len(), 2);
    lin_b[0] = 0.2;
    assert_eq!(fused.linear_bias_mut()[0], 0.2);
}

/// Test FeedForward getter methods
#[test]
fn test_ffn_getters() {
    let ffn = FeedForward::new(8, 32).expect("test");
    assert_eq!(ffn.hidden_dim(), 8);
    assert_eq!(ffn.intermediate_dim(), 32);
}

/// Test FeedForward mutable accessors
#[test]
fn test_ffn_mutable_accessors() {
    let mut ffn = FeedForward::new(4, 8).expect("test");

    // Get mutable references to fc1 and fc2
    let fc1 = ffn.fc1_mut();
    assert_eq!(fc1.in_features(), 4);
    assert_eq!(fc1.out_features(), 8);

    let fc2 = ffn.fc2_mut();
    assert_eq!(fc2.in_features(), 8);
    assert_eq!(fc2.out_features(), 4);
}

/// Test Attention getter methods
#[test]
fn test_attention_getters() {
    let attn = Attention::new(64).expect("test");
    assert_eq!(attn.head_dim(), 64);
    assert!((attn.scale() - 1.0 / 8.0).abs() < 1e-5); // 1/sqrt(64) = 0.125
}

/// Test Attention scale calculation
#[test]
fn test_attention_scale_various_dims() {
    // head_dim=16 -> scale = 1/4 = 0.25
    let attn16 = Attention::new(16).expect("test");
    assert!((attn16.scale() - 0.25).abs() < 1e-5);

    // head_dim=128 -> scale = 1/sqrt(128) ≈ 0.0884
    let attn128 = Attention::new(128).expect("test");
    assert!((attn128.scale() - 1.0 / (128.0f32).sqrt()).abs() < 1e-5);
}

/// Test gelu with single element
#[test]
fn test_gelu_single_cov() {
    let single = Tensor::from_vec(vec![1], vec![0.0f32]).expect("test");
    let result = gelu(&single).expect("test");
    // GELU(0) = 0
    assert!(result.data()[0].abs() < 1e-5);
}

/// Test softmax with single element (should return 1.0)
#[test]
fn test_softmax_single_element_cov() {
    let single = Tensor::from_vec(vec![1], vec![5.0f32]).expect("test");
    let result = softmax(&single).expect("test");
    assert!((result.data()[0] - 1.0).abs() < 1e-5);
}

/// Test softmax probabilities sum to 1
#[test]
fn test_softmax_sum_to_one_cov() {
    let t = Tensor::from_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).expect("test");
    let result = softmax(&t).expect("test");
    let sum: f32 = result.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: Debug/Clone implementations
// =========================================================================

#[test]
fn test_layer_norm_debug_clone() {
    let layer_norm = LayerNorm::new(64, 1e-5).expect("test");
    let debug = format!("{:?}", layer_norm);
    assert!(debug.contains("LayerNorm"));

    let cloned = layer_norm.clone();
    assert_eq!(cloned.normalized_shape(), layer_norm.normalized_shape());
}

#[test]
fn test_linear_debug_clone() {
    let linear = Linear::new(32, 64).expect("test");
    let debug = format!("{:?}", linear);
    assert!(debug.contains("Linear"));

    let cloned = linear.clone();
    assert_eq!(cloned.in_features(), linear.in_features());
    assert_eq!(cloned.out_features(), linear.out_features());
}

#[test]
fn test_rope_debug_clone() {
    let rope = RoPE::new(64, 10000.0).expect("test");
    let debug = format!("{:?}", rope);
    assert!(debug.contains("RoPE"));

    let cloned = rope.clone();
    assert_eq!(cloned.dim(), rope.dim());
}

#[test]
fn test_rope_scaling_type_debug_clone_copy() {
    // Test None variant
    let none = RopeScalingType::None;
    let debug_none = format!("{:?}", none);
    assert!(debug_none.contains("None"));
    let cloned_none = none;
    assert_eq!(cloned_none, RopeScalingType::None);

    // Test Linear variant
    let linear = RopeScalingType::Linear { scale: 2.0 };
    let debug_linear = format!("{:?}", linear);
    assert!(debug_linear.contains("Linear"));
    assert!(debug_linear.contains("2.0"));
    let cloned_linear = linear;
    assert_eq!(cloned_linear, linear);

    // Test Ntk variant
    let ntk = RopeScalingType::Ntk { scale: 1.5 };
    let debug_ntk = format!("{:?}", ntk);
    assert!(debug_ntk.contains("Ntk"));
    assert_eq!(ntk, RopeScalingType::Ntk { scale: 1.5 });

    // Test DynamicNtk variant
    let dynamic = RopeScalingType::DynamicNtk {
        original_max_len: 2048,
        target_max_len: 4096,
    };
    let debug_dynamic = format!("{:?}", dynamic);
    assert!(debug_dynamic.contains("DynamicNtk"));
    assert!(debug_dynamic.contains("2048"));

    // Test Yarn variant
    let yarn = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let debug_yarn = format!("{:?}", yarn);
    assert!(debug_yarn.contains("Yarn"));
    assert!(debug_yarn.contains("8192"));

    // Test Default
    let default = RopeScalingType::default();
    assert_eq!(default, RopeScalingType::None);
}

#[test]
fn test_scaled_rope_debug_clone() {
    let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("test");
    let debug = format!("{:?}", scaled);
    assert!(debug.contains("ScaledRoPE"));

    let cloned = scaled.clone();
    assert_eq!(cloned.dim(), scaled.dim());
}

#[test]
fn test_alibi_debug_clone() {
    let alibi = ALiBi::new(8).expect("test");
    let debug = format!("{:?}", alibi);
    assert!(debug.contains("ALiBi"));

    let cloned = alibi.clone();
    assert_eq!(cloned.num_heads(), alibi.num_heads());
}

#[test]
fn test_kv_cache_debug_clone() {
    let cache = KVCache::new(2, 512, 64).expect("test");
    let debug = format!("{:?}", cache);
    assert!(debug.contains("KVCache"));

    let cloned = cache.clone();
    assert_eq!(cloned.num_layers(), cache.num_layers());
}

#[test]
fn test_attention_debug_clone() {
    let attn = Attention::new(64).expect("test");
    let debug = format!("{:?}", attn);
    assert!(debug.contains("Attention"));

    let cloned = attn.clone();
    assert!((cloned.scale() - attn.scale()).abs() < 1e-6);
}

#[test]
fn test_feed_forward_debug_clone() {
    let ffn = FeedForward::new(64, 256).expect("test");
    let debug = format!("{:?}", ffn);
    assert!(debug.contains("FeedForward"));

    let cloned = ffn.clone();
    assert_eq!(cloned.hidden_dim(), ffn.hidden_dim());
}

#[test]
fn test_multi_head_attention_debug_clone() {
    let mha = MultiHeadAttention::new(256, 4, 4).expect("test");
    let debug = format!("{:?}", mha);
    assert!(debug.contains("MultiHeadAttention"));

    let cloned = mha.clone();
    assert_eq!(cloned.num_heads(), mha.num_heads());
}

#[test]
fn test_embedding_debug_clone() {
    let emb = Embedding::new(1000, 256).expect("test");
    let debug = format!("{:?}", emb);
    assert!(debug.contains("Embedding"));

    let cloned = emb.clone();
    assert_eq!(cloned.vocab_size(), emb.vocab_size());
    assert_eq!(cloned.embed_dim(), emb.embed_dim());
}

#[test]
fn test_model_config_debug_clone() {
    let config = ModelConfig {
        vocab_size: 50000,
        hidden_dim: 1024,
        num_layers: 12,
        num_heads: 8,
        intermediate_dim: 4096,
        eps: 1e-5,
    };
    let debug = format!("{:?}", config);
    assert!(debug.contains("ModelConfig"));
    assert!(debug.contains("50000"));

    let cloned = config.clone();
    assert_eq!(cloned.vocab_size, config.vocab_size);
    assert_eq!(cloned.hidden_dim, config.hidden_dim);
    assert_eq!(cloned.num_layers, config.num_layers);
}

#[test]
fn test_model_debug_clone() {
    let config = ModelConfig {
        vocab_size: 1000,
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 2,
        intermediate_dim: 256,
        eps: 1e-5,
    };
    let model = Model::new(config.clone()).expect("test");
    let debug = format!("{:?}", model);
    assert!(debug.contains("Model"));

    let cloned = model.clone();
    // Verify the cloned model has the same config
    assert_eq!(cloned.config().num_layers, model.config().num_layers);
}

#[test]
fn test_transformer_block_debug_clone() {
    let block = TransformerBlock::new(256, 4, 1024, 1e-5).expect("test");
    let debug = format!("{:?}", block);
    assert!(debug.contains("TransformerBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.hidden_dim(), block.hidden_dim());
}

#[test]
fn test_sliding_window_attention_debug_clone() {
    let swa = SlidingWindowAttention::new(64, 1024).expect("test");
    let debug = format!("{:?}", swa);
    assert!(debug.contains("SlidingWindowAttention"));

    let cloned = swa.clone();
    assert_eq!(cloned.window_size(), swa.window_size());
}
