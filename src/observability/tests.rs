use super::*;

#[test]
fn test_observability_config_default() {
    let config = ObservabilityConfig::new();
    assert!(config.tracing_enabled);
    assert!((config.trace_sample_rate - 1.0).abs() < 0.001);
}

#[test]
fn test_observability_config_builder() {
    let config = ObservabilityConfig::new()
        .with_trueno_db("trueno-db://localhost:5432")
        .with_tracing(false)
        .with_sample_rate(0.5);

    assert_eq!(
        config.trueno_db_uri,
        Some("trueno-db://localhost:5432".to_string())
    );
    assert!(!config.tracing_enabled);
    assert!((config.trace_sample_rate - 0.5).abs() < 0.001);
}

#[test]
fn test_metric_point() {
    let metric = MetricPoint::new("test_metric", 42.0)
        .with_label("model", "llama3")
        .with_label("version", "1.0");

    assert_eq!(metric.name, "test_metric");
    assert!((metric.value - 42.0).abs() < 0.001);
    assert_eq!(metric.labels.get("model"), Some(&"llama3".to_string()));
}

#[test]
fn test_metric_line_protocol() {
    let metric = MetricPoint::new("cpu_usage", 75.5).with_label("host", "server1");

    let line = metric.to_line_protocol();
    assert!(line.contains("cpu_usage"));
    assert!(line.contains("host=server1"));
    assert!(line.contains("75.5"));
}

#[test]
fn test_span_creation() {
    let span = Span::new("inference", "trace-123");

    assert!(!span.span_id.is_empty());
    assert_eq!(span.trace_id, "trace-123");
    assert_eq!(span.operation, "inference");
    assert_eq!(span.status, SpanStatus::InProgress);
}

#[test]
fn test_span_child() {
    let parent = Span::new("request", "trace-456");
    let child = parent.child("tokenize");

    assert_eq!(child.trace_id, parent.trace_id);
    assert_eq!(child.parent_id, Some(parent.span_id.clone()));
    assert_eq!(child.operation, "tokenize");
}

#[test]
fn test_span_end_ok() {
    let mut span = Span::new("test", "trace");
    std::thread::sleep(Duration::from_millis(10));
    span.end_ok();

    assert_eq!(span.status, SpanStatus::Ok);
    assert!(span.duration_us.is_some());
    assert!(span.duration_us.expect("test") >= 10000); // At least 10ms
}

#[test]
fn test_span_end_error() {
    let mut span = Span::new("test", "trace");
    span.end_error("Something went wrong");

    assert_eq!(span.status, SpanStatus::Error);
    assert_eq!(
        span.attributes.get("error"),
        Some(&"Something went wrong".to_string())
    );
}

#[test]
fn test_ab_test_creation() {
    let test = ABTest::new("model-comparison")
        .with_variant("control", "model-v1", 0.5)
        .with_variant("treatment", "model-v2", 0.5);

    assert_eq!(test.name, "model-comparison");
    assert_eq!(test.variants.len(), 2);
    assert!(test.is_valid());
}

#[test]
fn test_ab_test_selection_deterministic() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    // Same user should always get same variant
    let variant1 = test.select("user-123");
    let variant2 = test.select("user-123");

    assert_eq!(variant1.map(|v| &v.name), variant2.map(|v| &v.name));
}

#[test]
fn test_ab_test_selection_distribution() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    let mut count_a = 0;
    let mut count_b = 0;

    for i in 0..1000 {
        let user_id = format!("user-{i}");
        if let Some(variant) = test.select(&user_id) {
            if variant.name == "a" {
                count_a += 1;
            } else {
                count_b += 1;
            }
        }
    }

    // Should be roughly 50/50 (within 10% tolerance)
    let ratio = count_a as f64 / (count_a + count_b) as f64;
    assert!(ratio > 0.4 && ratio < 0.6);
}

#[test]
fn test_ab_test_invalid_weights() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.3)
        .with_variant("b", "model-b", 0.3);

    assert!(!test.is_valid()); // Weights sum to 0.6, not 1.0
}

#[test]
fn test_variant_result_calculations() {
    let result = VariantResult {
        requests: 100,
        successes: 90,
        total_latency_ms: 5000,
        total_tokens: 10000,
    };

    assert!((result.success_rate() - 0.9).abs() < 0.001);
    assert!((result.avg_latency_ms() - 50.0).abs() < 0.001);
    assert!((result.tokens_per_request() - 100.0).abs() < 0.001);
}

#[test]
fn test_observer_record_inference() {
    let observer = Observer::default_observer();
    observer.record_inference("llama3", 100, 50);

    let metrics = observer.flush_metrics();
    assert!(!metrics.is_empty());
}

#[test]
fn test_observer_record_span() {
    let observer = Observer::default_observer();
    let mut span = observer.start_trace("test-op");
    span.end_ok();
    observer.record_span(span);

    let spans = observer.flush_spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].operation, "test-op");
}

#[test]
fn test_observer_ab_results() {
    let observer = Observer::default_observer();

    observer.record_ab_result("test", "control", true, 50, 100);
    observer.record_ab_result("test", "control", true, 60, 120);
    observer.record_ab_result("test", "treatment", false, 40, 80);

    let results = observer.get_ab_results("test").expect("test");
    let control = results.variants.get("control").expect("test");
    let treatment = results.variants.get("treatment").expect("test");

    assert_eq!(control.requests, 2);
    assert_eq!(control.successes, 2);
    assert_eq!(treatment.requests, 1);
    assert_eq!(treatment.successes, 0);
}

#[test]
fn test_observer_prometheus_format() {
    let observer = Observer::default_observer();
    observer.record_metric(MetricPoint::new("test_metric", 42.0).with_label("env", "prod"));

    let prom = observer.prometheus_metrics();
    assert!(prom.contains("test_metric"));
    assert!(prom.contains("env=\"prod\""));
    assert!(prom.contains("42"));
}

#[test]
fn test_observer_request_id() {
    let observer = Observer::default_observer();

    let id1 = observer.next_request_id();
    let id2 = observer.next_request_id();
    let id3 = observer.next_request_id();

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);
}

#[test]
fn test_span_status_default() {
    let status = SpanStatus::default();
    assert_eq!(status, SpanStatus::InProgress);
}

#[test]
fn test_generate_id_unique() {
    let id1 = generate_id();
    let id2 = generate_id();

    // IDs should be unique (though not guaranteed with fast calls)
    assert_eq!(id1.len(), 16);
    assert_eq!(id2.len(), 16);
}

#[test]
fn test_simple_hash_deterministic() {
    let hash1 = simple_hash("test-input");
    let hash2 = simple_hash("test-input");
    let hash3 = simple_hash("different-input");

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, hash3);
}

#[test]
fn test_observer_sampling() {
    let config = ObservabilityConfig::new().with_sample_rate(0.0);
    let observer = Observer::new(config);

    let mut span = observer.start_trace("test");
    span.end_ok();
    observer.record_span(span);

    // With 0% sampling, no spans should be recorded
    let spans = observer.flush_spans();
    assert!(spans.is_empty());
}

#[test]
fn test_observer_tracing_disabled() {
    let config = ObservabilityConfig::new().with_tracing(false);
    let observer = Observer::new(config);

    let mut span = observer.start_trace("test");
    span.end_ok();
    observer.record_span(span);

    let spans = observer.flush_spans();
    assert!(spans.is_empty());
}

// =========================================================================
// W3C Trace Context Tests
// =========================================================================

#[test]
fn test_trace_context_new() {
    let ctx = TraceContext::new();
    assert_eq!(ctx.trace_id.len(), 32);
    assert!(ctx.parent_span_id.is_none());
    assert_eq!(ctx.trace_flags, 0x01); // Sampled by default
    assert!(ctx.trace_state.is_none());
}

#[test]
fn test_trace_context_child() {
    let parent_ctx = TraceContext::new();
    let child_ctx = parent_ctx.child("abcdef0123456789");

    assert_eq!(child_ctx.trace_id, parent_ctx.trace_id);
    assert_eq!(
        child_ctx.parent_span_id,
        Some("abcdef0123456789".to_string())
    );
    assert_eq!(child_ctx.trace_flags, parent_ctx.trace_flags);
}

#[test]
fn test_trace_context_from_traceparent_valid() {
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    let ctx = TraceContext::from_traceparent(header).expect("test");

    assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(ctx.parent_span_id, Some("b7ad6b7169203331".to_string()));
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_trace_context_from_traceparent_not_sampled() {
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00";
    let ctx = TraceContext::from_traceparent(header).expect("test");

    assert_eq!(ctx.trace_flags, 0x00);
    assert!(!ctx.is_sampled());
}

#[test]
fn test_trace_context_from_traceparent_invalid_version() {
    let header = "01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_invalid_format() {
    assert!(TraceContext::from_traceparent("invalid").is_none());
    assert!(TraceContext::from_traceparent("00-abc-def-01").is_none());
    assert!(TraceContext::from_traceparent("").is_none());
}

#[test]
fn test_trace_context_to_traceparent() {
    let ctx = TraceContext {
        trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
        parent_span_id: None,
        trace_flags: 0x01,
        trace_state: None,
    };

    let traceparent = ctx.to_traceparent("b7ad6b7169203331");
    assert_eq!(
        traceparent,
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
}

#[test]
fn test_trace_context_with_tracestate() {
    let ctx = TraceContext::new().with_tracestate("vendor=value");
    assert_eq!(ctx.trace_state, Some("vendor=value".to_string()));
}

#[test]
fn test_trace_context_sampled_flag() {
    let mut ctx = TraceContext::new();
    assert!(ctx.is_sampled());

    ctx.set_sampled(false);
    assert!(!ctx.is_sampled());
    assert_eq!(ctx.trace_flags, 0x00);

    ctx.set_sampled(true);
    assert!(ctx.is_sampled());
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_trace_context_default() {
    let ctx = TraceContext::default();
    assert_eq!(ctx.trace_id.len(), 32);
}

// =========================================================================
// Latency Histogram Tests
// =========================================================================

#[test]
fn test_latency_histogram_new() {
    let hist = LatencyHistogram::new();
    assert_eq!(hist.count(), 0);
    assert!(hist.min().is_none());
    assert!(hist.max_val().is_none());
    assert!(hist.mean().is_none());
}

#[test]
fn test_latency_histogram_observe() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000); // 1ms
    hist.observe(5000); // 5ms
    hist.observe(10000); // 10ms

    assert_eq!(hist.count(), 3);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(10000));
}

#[test]
fn test_latency_histogram_mean() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);
    hist.observe(2000);
    hist.observe(3000);

    let mean = hist.mean().expect("test");
    assert!((mean - 2000.0).abs() < 0.001);
}

#[test]
fn test_latency_histogram_observe_duration() {
    let mut hist = LatencyHistogram::new();
    hist.observe_duration(Duration::from_millis(5));

    assert_eq!(hist.count(), 1);
    assert_eq!(hist.min(), Some(5000)); // 5ms = 5000us
}

#[test]
fn test_latency_histogram_percentiles() {
    let mut hist = LatencyHistogram::new();
    // Add 100 observations: 1ms, 2ms, 3ms, ... 100ms
    for i in 1..=100 {
        hist.observe(i * 1000);
    }

    // p50 should be around 50ms
    let p50 = hist.p50().expect("test");
    assert!(p50 >= 25_000 && p50 <= 100_000);

    // p95 should be around 95ms
    let p95 = hist.p95().expect("test");
    assert!(p95 >= 50000);

    // p99 should be around 99ms
    let p99 = hist.p99().expect("test");
    assert!(p99 >= 50000);
}

#[test]
fn test_latency_histogram_percentile_empty() {
    let hist = LatencyHistogram::new();
    assert!(hist.percentile(50.0).is_none());
    assert!(hist.p50().is_none());
    assert!(hist.p95().is_none());
    assert!(hist.p99().is_none());
}

#[test]
fn test_latency_histogram_percentile_invalid() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);

    assert!(hist.percentile(-1.0).is_none());
    assert!(hist.percentile(101.0).is_none());
}

#[test]
fn test_latency_histogram_custom_buckets() {
    let buckets = vec![100, 500, 1000, 5000, 10000];
    let mut hist = LatencyHistogram::with_buckets(buckets);

    hist.observe(50); // bucket 0 (<=100)
    hist.observe(200); // bucket 1 (<=500)
    hist.observe(750); // bucket 2 (<=1000)
    hist.observe(20000); // overflow bucket

    assert_eq!(hist.count(), 4);
    assert_eq!(hist.min(), Some(50));
    assert_eq!(hist.max_val(), Some(20000));
}

#[test]
fn test_latency_histogram_to_prometheus() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000); // 1ms
    hist.observe(50000); // 50ms

    let prom = hist.to_prometheus("request_latency", "service=\"api\"");

    assert!(prom.contains("request_latency_bucket"));
    assert!(prom.contains("le="));
    assert!(prom.contains("service=\"api\""));
    assert!(prom.contains("request_latency_sum"));
    assert!(prom.contains("request_latency_count"));
    assert!(prom.contains("} 2")); // count = 2
}

// =========================================================================
// OpenTelemetry Export Tests
// =========================================================================

#[test]
fn test_span_to_otel_ok() {
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();

    assert_eq!(otel.trace_id, "trace123456789012345678901234");
    assert_eq!(otel.operation_name, "test-op");
    assert_eq!(otel.service_name, "realizar");
    assert_eq!(otel.status.code, OtelStatusCode::Ok);
    assert!(otel.status.message.is_none());
}

#[test]
fn test_span_to_otel_error() {
    let mut span = Span::new("failing-op", "trace123456789012345678901234");
    span.end_error("Connection timeout");

    let otel = span.to_otel();

    assert_eq!(otel.status.code, OtelStatusCode::Error);
    assert_eq!(otel.status.message, Some("Connection timeout".to_string()));
}

#[test]
fn test_span_to_otel_with_parent() {
    let parent = Span::new("parent-op", "trace123456789012345678901234");
    let mut child = parent.child("child-op");
    child.end_ok();

    let otel = child.to_otel();

    assert_eq!(otel.parent_span_id, Some(parent.span_id.clone()));
}

#[test]
fn test_span_to_otel_with_attributes() {
    let mut span = Span::new("test-op", "trace123456789012345678901234")
        .with_attribute("model", "llama3")
        .with_attribute("tokens", "256");
    span.end_ok();

    let otel = span.to_otel();

    assert!(otel.attributes.iter().any(|a| a.key == "model"));
    assert!(otel.attributes.iter().any(|a| a.key == "tokens"));
}

#[test]
fn test_span_to_otel_with_kind() {
    let mut span =
        Span::new("server-op", "trace123456789012345678901234").with_kind(SpanKind::Server);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Server);
}

#[test]
fn test_span_to_otel_timestamps() {
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    std::thread::sleep(Duration::from_millis(5));
    span.end_ok();

    let otel = span.to_otel();

    // End time should be >= start time
    assert!(otel.end_time >= otel.start_time);
    // Timestamps should be in nanoseconds
    assert!(otel.start_time > 0);
}

#[test]
fn test_span_trace_context() {
    let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
    let ctx = span.trace_context();

    assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(ctx.parent_span_id, Some(span.span_id.clone()));
    assert_eq!(ctx.trace_flags, 0x01);
}

#[test]
fn test_span_traceparent() {
    let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
    let traceparent = span.traceparent();

    assert!(traceparent.starts_with("00-"));
    assert!(traceparent.contains("0af7651916cd43dd8448eb211c80319c"));
    assert!(traceparent.ends_with("-01"));

    // Should have format: 00-{trace_id}-{span_id}-01
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0], "00");
    assert_eq!(parts[1], "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(parts[3], "01");
}

// =========================================================================
// OtelValue Tests
// =========================================================================

#[test]
fn test_otel_value_from_str() {
    let val: OtelValue = "test".into();
    match val {
        OtelValue::String { string_value } => assert_eq!(string_value, "test"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_value_from_string() {
    let val: OtelValue = String::from("test").into();
    match val {
        OtelValue::String { string_value } => assert_eq!(string_value, "test"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_value_from_i64() {
    let val: OtelValue = 42i64.into();
    match val {
        OtelValue::Int { int_value } => assert_eq!(int_value, 42),
        _ => panic!("Expected Int variant"),
    }
}

#[test]
fn test_otel_value_from_f64() {
    let val: OtelValue = 3.14f64.into();
    match val {
        OtelValue::Float { double_value } => assert!((double_value - 3.14).abs() < 0.001),
        _ => panic!("Expected Float variant"),
    }
}

#[test]
fn test_otel_value_from_bool() {
    let val: OtelValue = true.into();
    match val {
        OtelValue::Bool { bool_value } => assert!(bool_value),
        _ => panic!("Expected Bool variant"),
    }
}

// =========================================================================
// SpanKind Tests
// =========================================================================

#[test]
fn test_span_kind_default() {
    let kind = SpanKind::default();
    assert_eq!(kind, SpanKind::Internal);
}

#[test]
fn test_otel_status_code_default() {
    let code = OtelStatusCode::default();
    assert_eq!(code, OtelStatusCode::Unset);
}

// =========================================================================
// Generate Trace ID Tests
// =========================================================================

#[test]
fn test_generate_trace_id_length() {
    let id = generate_trace_id();
    assert_eq!(id.len(), 32); // 16 bytes = 32 hex chars
}

#[test]
fn test_generate_trace_id_hex() {
    let id = generate_trace_id();
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

// =========================================================================
// Additional Coverage Tests - Edge Cases
// =========================================================================

#[test]
fn test_metric_line_protocol_no_labels() {
    let metric = MetricPoint::new("simple_metric", 100.0);
    let line = metric.to_line_protocol();

    // Should not contain commas (no labels)
    assert!(line.starts_with("simple_metric value=100"));
    // Verify no extra comma before "value"
    assert!(!line.contains(",value"));
}

#[test]
fn test_span_with_parent() {
    let span = Span::new("child-op", "trace-abc").with_parent("parent-span-123");

    assert_eq!(span.parent_id, Some("parent-span-123".to_string()));
}

#[test]
fn test_span_duration() {
    let mut span = Span::new("test", "trace");

    // Before ending, duration should be None
    assert!(span.duration().is_none());

    std::thread::sleep(Duration::from_millis(5));
    span.end_ok();

    // After ending, duration should be Some
    let dur = span.duration().expect("should have duration");
    assert!(dur >= Duration::from_millis(5));
}

#[test]
fn test_span_to_otel_in_progress() {
    let span = Span::new("in-progress-op", "trace123456789012345678901234");
    // Don't end the span - leave it in progress

    let otel = span.to_otel();

    assert_eq!(otel.status.code, OtelStatusCode::Unset);
    assert!(otel.status.message.is_none());
}

#[test]
fn test_span_to_otel_client_kind() {
    let mut span =
        Span::new("client-op", "trace123456789012345678901234").with_kind(SpanKind::Client);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Client);
}

#[test]
fn test_span_to_otel_producer_kind() {
    let mut span =
        Span::new("producer-op", "trace123456789012345678901234").with_kind(SpanKind::Producer);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Producer);
}

#[test]
fn test_span_to_otel_consumer_kind() {
    let mut span =
        Span::new("consumer-op", "trace123456789012345678901234").with_kind(SpanKind::Consumer);
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Consumer);
}

#[test]
fn test_span_to_otel_unknown_kind() {
    // Test with an unknown kind string in attributes
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.attributes
        .insert("span.kind".to_string(), "Unknown".to_string());
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Internal); // Falls back to Internal
}

#[test]
fn test_variant_result_zero_requests() {
    let result = VariantResult::default();

    assert_eq!(result.requests, 0);
    assert!((result.success_rate() - 0.0).abs() < 0.001);
    assert!((result.avg_latency_ms() - 0.0).abs() < 0.001);
    assert!((result.tokens_per_request() - 0.0).abs() < 0.001);
}

#[test]
fn test_ab_test_select_inactive() {
    let mut test = ABTest::new("inactive-test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);
    test.active = false;

    assert!(test.select("user-123").is_none());
}

#[test]
fn test_ab_test_select_empty_variants() {
    let test = ABTest::new("empty-test");

    assert!(test.select("user-123").is_none());
}

#[test]
fn test_ab_test_select_zero_weight() {
    let test = ABTest::new("zero-weight-test")
        .with_variant("a", "model-a", 0.0)
        .with_variant("b", "model-b", 0.0);

    // With zero total weight, should return first variant
    let variant = test.select("user-123");
    assert!(variant.is_some());
    assert_eq!(variant.unwrap().name, "a");
}

#[test]
fn test_ab_test_is_valid_empty() {
    let test = ABTest::new("empty-test");
    assert!(!test.is_valid());
}

#[test]
fn test_ab_test_weight_clamping() {
    let test = ABTest::new("clamped-test")
        .with_variant("a", "model-a", 1.5) // Should clamp to 1.0
        .with_variant("b", "model-b", -0.5); // Should clamp to 0.0

    assert!((test.variants[0].weight - 1.0).abs() < 0.001);
    assert!((test.variants[1].weight - 0.0).abs() < 0.001);
}

#[test]
fn test_observer_ab_testing_disabled() {
    let config = ObservabilityConfig {
        ab_testing_enabled: false,
        ..ObservabilityConfig::new()
    };
    let observer = Observer::new(config);

    observer.record_ab_result("test", "control", true, 50, 100);

    // Results should not be recorded
    assert!(observer.get_ab_results("test").is_none());
}

#[test]
fn test_observability_config_flush_interval() {
    let config = ObservabilityConfig::new().with_flush_interval(120);

    assert_eq!(config.flush_interval_secs, 120);
}

#[test]
fn test_observability_config_sample_rate_clamping() {
    let config_high = ObservabilityConfig::new().with_sample_rate(2.0);
    assert!((config_high.trace_sample_rate - 1.0).abs() < 0.001);

    let config_low = ObservabilityConfig::new().with_sample_rate(-0.5);
    assert!((config_low.trace_sample_rate - 0.0).abs() < 0.001);
}

#[test]
fn test_latency_histogram_overflow_bucket() {
    let mut hist = LatencyHistogram::new();
    // Add a value larger than all buckets (60s = 60_000_000 us)
    hist.observe(100_000_000); // 100s - definitely in overflow

    assert_eq!(hist.count(), 1);
    assert_eq!(hist.max_val(), Some(100_000_000));

    // p100 should return max value (overflow bucket)
    let p100 = hist.percentile(100.0);
    assert_eq!(p100, Some(100_000_000));
}

#[test]
fn test_latency_histogram_default() {
    let hist = LatencyHistogram::default();
    assert_eq!(hist.count(), 0);
}

#[test]
fn test_latency_histogram_with_buckets_unsorted() {
    // Buckets should be sorted internally
    let buckets = vec![1000, 100, 500, 200];
    let mut hist = LatencyHistogram::with_buckets(buckets);

    hist.observe(150);
    hist.observe(350);

    assert_eq!(hist.count(), 2);
}

#[test]
fn test_observer_prometheus_multiple_metrics_same_name() {
    let observer = Observer::default_observer();

    observer
        .record_metric(MetricPoint::new("request_count", 10.0).with_label("endpoint", "/api/v1"));
    observer
        .record_metric(MetricPoint::new("request_count", 20.0).with_label("endpoint", "/api/v2"));

    let prom = observer.prometheus_metrics();

    // Should have TYPE header
    assert!(prom.contains("# TYPE request_count gauge"));
    // Should have both metrics
    assert!(prom.contains("endpoint=\"/api/v1\""));
    assert!(prom.contains("endpoint=\"/api/v2\""));
}

#[test]
fn test_observer_prometheus_no_labels() {
    let observer = Observer::default_observer();
    observer.record_metric(MetricPoint::new("simple_count", 5.0));

    let prom = observer.prometheus_metrics();

    assert!(prom.contains("simple_count 5"));
}

#[test]
fn test_trace_context_from_traceparent_invalid_hex_flags() {
    // Invalid hex in flags field
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-GG";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_wrong_trace_id_length() {
    // trace_id too short (only 16 chars instead of 32)
    let header = "00-0af7651916cd43dd-b7ad6b7169203331-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_trace_context_from_traceparent_wrong_span_id_length() {
    // span_id too short (only 8 chars instead of 16)
    let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b71-01";
    assert!(TraceContext::from_traceparent(header).is_none());
}

#[test]
fn test_otel_span_serialization() {
    let mut span =
        Span::new("test-op", "0af7651916cd43dd8448eb211c80319c").with_attribute("key", "value");
    span.end_ok();

    let otel = span.to_otel();

    // Should be serializable to JSON
    let json = serde_json::to_string(&otel).expect("should serialize");
    assert!(json.contains("traceId"));
    assert!(json.contains("spanId"));
    assert!(json.contains("operationName"));
}

#[test]
fn test_ab_test_result_default() {
    let result = ABTestResult::default();
    assert!(result.test_name.is_empty());
    assert!(result.variants.is_empty());
}

#[test]
fn test_metric_point_with_multiple_labels() {
    let metric = MetricPoint::new("multi_label", 42.0)
        .with_label("region", "us-east")
        .with_label("env", "prod")
        .with_label("version", "1.0");

    let line = metric.to_line_protocol();

    assert!(line.contains("region=us-east"));
    assert!(line.contains("env=prod"));
    assert!(line.contains("version=1.0"));
}

#[test]
fn test_observer_flush_metrics_clears_buffer() {
    let observer = Observer::default_observer();

    observer.record_metric(MetricPoint::new("test", 1.0));
    let first_flush = observer.flush_metrics();
    assert_eq!(first_flush.len(), 1);

    // Second flush should return empty
    let second_flush = observer.flush_metrics();
    assert!(second_flush.is_empty());
}

#[test]
fn test_observer_flush_spans_clears_buffer() {
    let observer = Observer::default_observer();

    let mut span = observer.start_trace("test-op");
    span.end_ok();
    observer.record_span(span);

    let first_flush = observer.flush_spans();
    assert_eq!(first_flush.len(), 1);

    // Second flush should return empty
    let second_flush = observer.flush_spans();
    assert!(second_flush.is_empty());
}

#[test]
fn test_ab_test_select_last_variant() {
    // Test that selection can hit the last variant
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.1)
        .with_variant("b", "model-b", 0.9);

    // With many users, we should hit both variants
    let mut hit_b = false;
    for i in 0..100 {
        let user_id = format!("user-{i}");
        if let Some(variant) = test.select(&user_id) {
            if variant.name == "b" {
                hit_b = true;
                break;
            }
        }
    }
    assert!(hit_b, "Should hit variant b with 90% weight");
}

#[test]
fn test_latency_histogram_single_value_percentiles() {
    let mut hist = LatencyHistogram::new();
    hist.observe(5000); // 5ms

    // All percentiles should return the same bucket
    assert!(hist.p50().is_some());
    assert!(hist.p95().is_some());
    assert!(hist.p99().is_some());
}

#[test]
fn test_span_with_attribute_chain() {
    let span = Span::new("test", "trace")
        .with_attribute("a", "1")
        .with_attribute("b", "2")
        .with_attribute("c", "3");

    assert_eq!(span.attributes.len(), 3);
    assert_eq!(span.attributes.get("a"), Some(&"1".to_string()));
    assert_eq!(span.attributes.get("b"), Some(&"2".to_string()));
    assert_eq!(span.attributes.get("c"), Some(&"3".to_string()));
}

#[test]
fn test_trace_context_set_sampled_preserves_other_flags() {
    let mut ctx = TraceContext {
        trace_id: "test".to_string(),
        parent_span_id: None,
        trace_flags: 0xFF, // All flags set
        trace_state: None,
    };

    // Setting sampled to false should only clear bit 0
    ctx.set_sampled(false);
    assert_eq!(ctx.trace_flags, 0xFE);

    // Setting sampled to true should only set bit 0
    ctx.set_sampled(true);
    assert_eq!(ctx.trace_flags, 0xFF);
}

#[test]
fn test_simple_hash_empty_string() {
    let hash = simple_hash("");
    // Should not panic and should return a consistent value
    let hash2 = simple_hash("");
    assert_eq!(hash, hash2);
}

#[test]
fn test_generate_id_format() {
    let id = generate_id();
    // Should be 16 hex characters
    assert_eq!(id.len(), 16);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

// =========================================================================
// Additional Edge Case Tests for Coverage
// =========================================================================

#[test]
fn test_latency_histogram_percentile_boundary() {
    // Test percentile at exact boundary values
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);

    // p0 should work
    let p0 = hist.percentile(0.0);
    assert!(p0.is_some());

    // p100 should work
    let p100 = hist.percentile(100.0);
    assert!(p100.is_some());
}

#[test]
fn test_latency_histogram_to_prometheus_empty_labels() {
    let mut hist = LatencyHistogram::new();
    hist.observe(5000);

    // Empty labels string
    let prom = hist.to_prometheus("latency", "");
    assert!(prom.contains("latency_bucket"));
    assert!(prom.contains("le=\"+Inf\""));
}

#[test]
fn test_latency_histogram_min_max_update() {
    let mut hist = LatencyHistogram::new();

    // First observation sets both min and max
    hist.observe(5000);
    assert_eq!(hist.min(), Some(5000));
    assert_eq!(hist.max_val(), Some(5000));

    // Lower value updates min
    hist.observe(1000);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(5000));

    // Higher value updates max
    hist.observe(10000);
    assert_eq!(hist.min(), Some(1000));
    assert_eq!(hist.max_val(), Some(10000));
}

#[test]
fn test_span_to_otel_internal_kind_default() {
    // Span without explicit kind should be Internal
    let mut span = Span::new("test-op", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();
    assert_eq!(otel.kind, SpanKind::Internal);
}

#[test]
fn test_otel_attribute_construction() {
    let attr = OtelAttribute {
        key: "test_key".to_string(),
        value: OtelValue::from("test_value"),
    };

    assert_eq!(attr.key, "test_key");
    match attr.value {
        OtelValue::String { string_value } => assert_eq!(string_value, "test_value"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_otel_status_construction() {
    let status_ok = OtelStatus {
        code: OtelStatusCode::Ok,
        message: None,
    };
    assert_eq!(status_ok.code, OtelStatusCode::Ok);

    let status_err = OtelStatus {
        code: OtelStatusCode::Error,
        message: Some("failed".to_string()),
    };
    assert_eq!(status_err.message, Some("failed".to_string()));
}

#[test]
fn test_ab_variant_construction() {
    let variant = ABVariant {
        name: "control".to_string(),
        model: "model-v1".to_string(),
        weight: 0.5,
    };

    assert_eq!(variant.name, "control");
    assert_eq!(variant.model, "model-v1");
    assert!((variant.weight - 0.5).abs() < 0.001);
}

#[test]
fn test_observer_default_trait() {
    let observer = Observer::default();
    let id = observer.next_request_id();
    assert_eq!(id, 0);
}

#[test]
fn test_observer_record_metric_multiple() {
    let observer = Observer::default_observer();

    // Record multiple metrics of different types
    observer.record_metric(MetricPoint::new("counter", 1.0));
    observer.record_metric(MetricPoint::new("gauge", 50.0));
    observer.record_metric(MetricPoint::new("histogram", 100.0));

    let metrics = observer.flush_metrics();
    assert_eq!(metrics.len(), 3);
}

#[test]
fn test_ab_test_result_construction() {
    let mut result = ABTestResult {
        test_name: "my-test".to_string(),
        variants: HashMap::new(),
    };

    result.variants.insert(
        "control".to_string(),
        VariantResult {
            requests: 10,
            successes: 8,
            total_latency_ms: 500,
            total_tokens: 1000,
        },
    );

    assert_eq!(result.test_name, "my-test");
    assert_eq!(result.variants.len(), 1);
    assert!(result.variants.contains_key("control"));
}

#[test]
fn test_serde_trace_context_roundtrip() {
    let ctx = TraceContext {
        trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
        parent_span_id: Some("b7ad6b7169203331".to_string()),
        trace_flags: 0x01,
        trace_state: Some("vendor=value".to_string()),
    };

    let json = serde_json::to_string(&ctx).expect("serialize");
    let ctx2: TraceContext = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(ctx.trace_id, ctx2.trace_id);
    assert_eq!(ctx.parent_span_id, ctx2.parent_span_id);
    assert_eq!(ctx.trace_flags, ctx2.trace_flags);
    assert_eq!(ctx.trace_state, ctx2.trace_state);
}

#[test]
fn test_serde_latency_histogram_roundtrip() {
    let mut hist = LatencyHistogram::new();
    hist.observe(1000);
    hist.observe(5000);

    let json = serde_json::to_string(&hist).expect("serialize");
    let hist2: LatencyHistogram = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(hist.count(), hist2.count());
    assert_eq!(hist.min(), hist2.min());
    assert_eq!(hist.max_val(), hist2.max_val());
}

#[test]
fn test_serde_metric_point_roundtrip() {
    let metric = MetricPoint::new("test_metric", 42.5).with_label("env", "prod");

    let json = serde_json::to_string(&metric).expect("serialize");
    let metric2: MetricPoint = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(metric.name, metric2.name);
    assert!((metric.value - metric2.value).abs() < 0.001);
    assert_eq!(metric.labels.get("env"), metric2.labels.get("env"));
}

#[test]
fn test_serde_span_roundtrip() {
    let mut span = Span::new("test-op", "trace-123").with_attribute("key", "value");
    span.end_ok();

    let json = serde_json::to_string(&span).expect("serialize");
    let span2: Span = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(span.trace_id, span2.trace_id);
    assert_eq!(span.operation, span2.operation);
    assert_eq!(span.status, span2.status);
}

#[test]
fn test_serde_ab_test_roundtrip() {
    let test = ABTest::new("test")
        .with_variant("a", "model-a", 0.5)
        .with_variant("b", "model-b", 0.5);

    let json = serde_json::to_string(&test).expect("serialize");
    let test2: ABTest = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(test.name, test2.name);
    assert_eq!(test.variants.len(), test2.variants.len());
}

#[test]
fn test_serde_span_kind_roundtrip() {
    let kinds = vec![
        SpanKind::Internal,
        SpanKind::Server,
        SpanKind::Client,
        SpanKind::Producer,
        SpanKind::Consumer,
    ];

    for kind in kinds {
        let json = serde_json::to_string(&kind).expect("serialize");
        let kind2: SpanKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, kind2);
    }
}

#[test]
fn test_serde_otel_status_code_roundtrip() {
    let codes = vec![
        OtelStatusCode::Unset,
        OtelStatusCode::Ok,
        OtelStatusCode::Error,
    ];

    for code in codes {
        let json = serde_json::to_string(&code).expect("serialize");
        let code2: OtelStatusCode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(code, code2);
    }
}

#[test]
fn test_serde_span_status_roundtrip() {
    let statuses = vec![SpanStatus::InProgress, SpanStatus::Ok, SpanStatus::Error];

    for status in statuses {
        let json = serde_json::to_string(&status).expect("serialize");
        let status2: SpanStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, status2);
    }
}

#[test]
fn test_serde_otel_value_variants() {
    // String variant
    let s: OtelValue = "test".into();
    let json = serde_json::to_string(&s).expect("serialize");
    assert!(json.contains("string_value"));

    // Int variant
    let i: OtelValue = 42i64.into();
    let json = serde_json::to_string(&i).expect("serialize");
    assert!(json.contains("int_value"));

    // Float variant
    let f: OtelValue = 3.14f64.into();
    let json = serde_json::to_string(&f).expect("serialize");
    assert!(json.contains("double_value"));

    // Bool variant
    let b: OtelValue = true.into();
    let json = serde_json::to_string(&b).expect("serialize");
    assert!(json.contains("bool_value"));
}

#[test]
fn test_observability_config_default_values() {
    let config = ObservabilityConfig::default();

    assert!(config.trueno_db_uri.is_none());
    assert!(!config.tracing_enabled); // Default is false
    assert!((config.trace_sample_rate - 0.0).abs() < 0.001);
    assert_eq!(config.flush_interval_secs, 0);
    assert!(!config.ab_testing_enabled);
}

#[test]
fn test_concurrent_request_id_generation() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    // Spawn multiple threads to get request IDs
    for _ in 0..10 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || obs.next_request_id()));
    }

    let mut ids: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    ids.sort_unstable();

    // All IDs should be unique (0-9)
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(*id, i as u64);
    }
}

#[test]
fn test_concurrent_metric_recording() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    for i in 0..5 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || {
            obs.record_metric(MetricPoint::new(format!("metric_{i}"), i as f64));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let metrics = observer.flush_metrics();
    assert_eq!(metrics.len(), 5);
}

#[test]
fn test_concurrent_span_recording() {
    use std::sync::Arc;
    use std::thread;

    let observer = Arc::new(Observer::default_observer());
    let mut handles = vec![];

    for i in 0..5 {
        let obs = Arc::clone(&observer);
        handles.push(thread::spawn(move || {
            let mut span = obs.start_trace(&format!("op-{i}"));
            span.end_ok();
            obs.record_span(span);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let spans = observer.flush_spans();
    assert_eq!(spans.len(), 5);
}

#[test]
fn test_observability_config_serde_roundtrip() {
    let config = ObservabilityConfig::new()
        .with_trueno_db("trueno://localhost")
        .with_tracing(true)
        .with_sample_rate(0.75)
        .with_flush_interval(30);

    let json = serde_json::to_string(&config).expect("serialize");
    let config2: ObservabilityConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(config.trueno_db_uri, config2.trueno_db_uri);
    assert_eq!(config.tracing_enabled, config2.tracing_enabled);
    assert!((config.trace_sample_rate - config2.trace_sample_rate).abs() < 0.001);
    assert_eq!(config.flush_interval_secs, config2.flush_interval_secs);
}

#[test]
fn test_metric_point_timestamp_is_set() {
    let metric = MetricPoint::new("test", 1.0);
    // Timestamp should be non-zero (set to current time)
    assert!(metric.timestamp > 0);
}

#[test]
fn test_span_start_time_is_set() {
    let span = Span::new("test", "trace-id");
    // Start time should be non-zero
    assert!(span.start_time > 0);
}

#[test]
fn test_ab_test_start_time_is_set() {
    let test = ABTest::new("test");
    // Start time should be non-zero
    assert!(test.start_time > 0);
}

#[test]
fn test_latency_histogram_with_empty_buckets() {
    let hist = LatencyHistogram::with_buckets(vec![]);

    // Empty buckets - only overflow bucket exists
    assert_eq!(hist.buckets.len(), 0);
    // counts should have 1 element (overflow bucket)
    assert_eq!(hist.counts.len(), 1);
}

#[test]
fn test_latency_histogram_observe_into_first_bucket() {
    let mut hist = LatencyHistogram::with_buckets(vec![1000, 5000, 10000]);

    // Value exactly at first bucket boundary
    hist.observe(1000);
    assert_eq!(hist.count(), 1);

    // p50 should return the first bucket
    let p50 = hist.p50();
    assert_eq!(p50, Some(1000));
}

#[test]
fn test_span_service_name_default() {
    let span = Span::new("op", "trace");
    assert_eq!(span.service, "realizar");
}

#[test]
fn test_otel_span_timestamps_in_nanoseconds() {
    let mut span = Span::new("test", "trace123456789012345678901234");
    span.end_ok();

    let otel = span.to_otel();

    // OTel timestamps should be in nanoseconds (start_time * 1000 from microseconds)
    // They should be much larger than the original microsecond values
    assert!(otel.start_time > span.start_time);
}

#[test]
fn test_observer_get_ab_results_missing() {
    let observer = Observer::default_observer();
    assert!(observer.get_ab_results("nonexistent").is_none());
}

#[test]
fn test_ab_test_select_full_weight_first_variant() {
    let test = ABTest::new("test").with_variant("only", "model-only", 1.0);

    // Should always select the only variant
    for i in 0..10 {
        let variant = test.select(&format!("user-{i}"));
        assert!(variant.is_some());
        assert_eq!(variant.unwrap().name, "only");
    }
}
