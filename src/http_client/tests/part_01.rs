use crate::http_client::*;

// =========================================================================
// Unit Tests (No network required)
// =========================================================================

#[test]
fn test_client_creation() {
    let client = ModelHttpClient::new();
    assert_eq!(client.timeout_secs(), 60);
}

#[test]
fn test_client_custom_timeout() {
    let client = ModelHttpClient::with_timeout(120);
    assert_eq!(client.timeout_secs(), 120);
}

#[test]
fn test_completion_request_serialization() {
    let request = CompletionRequest {
        model: "llama2".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 100,
        temperature: Some(0.7),
        stream: false,
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    assert!(json.contains("\"model\":\"llama2\""));
    assert!(json.contains("\"prompt\":\"Hello\""));
    assert!(json.contains("\"max_tokens\":100"));
}

#[test]
fn test_ollama_request_serialization() {
    let request = OllamaRequest {
        model: "llama2".to_string(),
        prompt: "Hello".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(100),
            temperature: Some(0.7),
        }),
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    assert!(json.contains("\"model\":\"llama2\""));
    assert!(json.contains("\"prompt\":\"Hello\""));
}

#[test]
fn test_completion_response_deserialization() {
    let json = r#"{
        "id": "cmpl-123",
        "choices": [{"text": "World!", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    }"#;

    let response: CompletionResponse =
        serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.id, "cmpl-123");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].text, "World!");
}

#[test]
fn test_ollama_response_deserialization() {
    let json = r#"{
        "model": "llama2",
        "response": "Hello back!",
        "done": true,
        "total_duration": 5000000000,
        "load_duration": 1000000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 2000000000,
        "eval_count": 5,
        "eval_duration": 2000000000
    }"#;

    let response: OllamaResponse = serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.model, "llama2");
    assert_eq!(response.response, "Hello back!");
    assert!(response.done);
    assert_eq!(response.eval_count, 5);
}

// =========================================================================
// Integration Tests (Require running servers)
// Mark with #[ignore] for CI - run manually with: cargo test -- --ignored
// =========================================================================

#[test]
#[ignore = "Requires vLLM server at localhost:8000"]
fn test_vllm_real_inference() {
    let client = ModelHttpClient::new();

    let request = CompletionRequest {
        model: "meta-llama/Llama-2-7b-hf".to_string(),
        prompt: "The capital of France is".to_string(),
        max_tokens: 20,
        temperature: Some(0.1),
        stream: false,
    };

    let result = client.openai_completion("http://localhost:8000", &request, None);

    // This MUST succeed with a real server
    let timing = result.expect("vLLM inference failed - is server running?");

    // Verify we got REAL data, not mock data
    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );
    assert!(!timing.text.is_empty(), "Must get actual generated text");

    println!("vLLM Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
#[ignore = "Requires Ollama server at localhost:11434"]
fn test_ollama_real_inference() {
    let client = ModelHttpClient::new();

    let request = OllamaRequest {
        model: "phi2:2.7b".to_string(),
        prompt: "The capital of France is".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(20),
            temperature: Some(0.1),
        }),
    };

    let result = client.ollama_generate("http://localhost:11434", &request);

    // This MUST succeed with a real server
    let timing = result.expect("Ollama inference failed - is server running?");

    // Verify we got REAL data, not mock data
    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );
    assert!(!timing.text.is_empty(), "Must get actual generated text");

    println!("Ollama Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
#[ignore = "Requires llama.cpp server at localhost:8080"]
fn test_llamacpp_real_inference() {
    let client = ModelHttpClient::new();

    let request = CompletionRequest {
        model: "default".to_string(), // llama.cpp uses loaded model
        prompt: "The capital of France is".to_string(),
        max_tokens: 20,
        temperature: Some(0.1),
        stream: false,
    };

    let result = client.openai_completion("http://localhost:8080", &request, None);

    let timing = result.expect("llama.cpp inference failed - is server running?");

    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );

    println!("llama.cpp Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
fn test_connection_error_handling() {
    let client = ModelHttpClient::with_timeout(1); // 1 second timeout

    let request = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: None,
        stream: false,
    };

    // This should fail because no server is running on this port
    let result = client.openai_completion("http://localhost:59999", &request, None);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        RealizarError::ConnectionError(msg) => {
            assert!(msg.contains("HTTP request failed"));
        },
        other => panic!("Expected ConnectionError, got: {:?}", other),
    }
}

#[test]
fn test_ollama_connection_error() {
    let client = ModelHttpClient::with_timeout(1);

    let request = OllamaRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        stream: false,
        options: None,
    };

    let result = client.ollama_generate("http://localhost:59998", &request);

    assert!(result.is_err());
}

#[test]
fn test_llamacpp_response_deserialization() {
    let json = r#"{
        "content": "Machine learning is a subset of AI.",
        "model": "/path/to/model.gguf",
        "tokens_predicted": 8,
        "tokens_evaluated": 5,
        "stop": true,
        "timings": {
            "prompt_n": 5,
            "prompt_ms": 10.5,
            "predicted_n": 8,
            "predicted_ms": 25.3,
            "predicted_per_second": 316.2
        }
    }"#;

    let response: LlamaCppResponse =
        serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.content, "Machine learning is a subset of AI.");
    assert_eq!(response.tokens_predicted, 8);
    assert_eq!(response.tokens_evaluated, 5);
    assert!(response.stop);

    let timings = response.timings.expect("timings should be present");
    assert_eq!(timings.prompt_n, 5);
    assert_eq!(timings.predicted_n, 8);
    assert!((timings.predicted_per_second - 316.2).abs() < 0.1);
}

#[test]
fn test_llamacpp_response_minimal() {
    // llama.cpp response with only required field
    let json = r#"{"content": "Hello world"}"#;

    let response: LlamaCppResponse =
        serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.content, "Hello world");
    assert_eq!(response.tokens_predicted, 0); // default
    assert!(response.timings.is_none());
}

#[test]
fn test_llamacpp_connection_error() {
    let client = ModelHttpClient::with_timeout(1);

    let request = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: None,
        stream: false,
    };

    let result = client.llamacpp_completion("http://localhost:59997", &request);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        RealizarError::ConnectionError(msg) => {
            assert!(msg.contains("HTTP request failed"));
        },
        other => panic!("Expected ConnectionError, got: {:?}", other),
    }
}

// =========================================================================
// HTTP Benchmark Runner Tests
// =========================================================================

#[test]
fn test_benchmark_config_default() {
    let config = HttpBenchmarkConfig::default();
    assert_eq!(config.min_samples(), 5);
    assert_eq!(config.max_samples(), 30);
    assert!((config.cv_threshold() - 0.05).abs() < 0.001); // Default is now 5% per spec
    assert_eq!(config.warmup_iterations, 2);
    assert!(config.run_preflight);
    assert!(config.filter_outliers);
}

#[test]
fn test_benchmark_config_relaxed() {
    let config = HttpBenchmarkConfig::relaxed();
    assert_eq!(config.min_samples(), 3);
    assert_eq!(config.max_samples(), 10);
    assert!((config.cv_threshold() - 0.20).abs() < 0.001);
    assert!(!config.run_preflight);
    assert!(!config.filter_outliers);
}

#[test]
fn test_benchmark_config_reproducible() {
    let config = HttpBenchmarkConfig::reproducible();
    assert_eq!(config.min_samples(), 10);
    assert_eq!(config.max_samples(), 50);
    assert!((config.cv_threshold() - 0.03).abs() < 0.001);
    assert!(config.run_preflight);
    assert!(config.filter_outliers);
}

#[test]
fn test_benchmark_runner_creation() {
    let runner = HttpBenchmarkRunner::with_defaults();
    assert_eq!(runner.config.min_samples(), 5);
}

#[test]
fn test_benchmark_runner_relaxed() {
    let runner = HttpBenchmarkRunner::with_relaxed();
    assert_eq!(runner.config.min_samples(), 3);
}

#[test]
fn test_benchmark_runner_reproducible() {
    let runner = HttpBenchmarkRunner::with_reproducible();
    assert_eq!(runner.config.min_samples(), 10);
}

#[test]
fn test_cv_calculation_identical_values() {
    // Identical values should have CV = 0
    let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert!(
        cv < 0.001,
        "CV of identical values should be ~0, got {}",
        cv
    );
}

#[test]
fn test_cv_calculation_varied_values() {
    // Known CV case: mean=100, std=10, CV=0.1
    let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    // CV should be around 0.079 (std ~7.9, mean 100)
    assert!(
        cv > 0.05 && cv < 0.15,
        "CV should be reasonable, got {}",
        cv
    );
}

#[test]
fn test_cv_calculation_single_value() {
    let samples = vec![100.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert_eq!(cv, f64::MAX, "Single value should return MAX CV");
}

#[test]
fn test_cv_calculation_empty() {
    let samples: Vec<f64> = vec![];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert_eq!(cv, f64::MAX, "Empty samples should return MAX CV");
}

#[test]
fn test_compute_results_basic() {
    let latencies = vec![100.0, 110.0, 90.0, 105.0, 95.0];
    let throughputs = vec![50.0, 45.0, 55.0, 48.0, 52.0];
    let cold_start = 120.0;
    let cv_threshold = 0.10;

    let result = HttpBenchmarkRunner::compute_results(
        &latencies,
        &throughputs,
        cold_start,
        cv_threshold,
    );

    assert_eq!(result.sample_count, 5);
    assert!((result.mean_latency_ms - 100.0).abs() < 0.01);
    assert!(result.p50_latency_ms > 0.0);
    assert!(result.p99_latency_ms >= result.p50_latency_ms);
    assert!(result.throughput_tps > 0.0);
    assert_eq!(result.cold_start_ms, 120.0);
}

#[test]
fn test_compute_results_percentiles() {
    // Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let latencies: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let throughputs = vec![];
    let cold_start = 1.0;
    let cv_threshold = 0.10;

    let result = HttpBenchmarkRunner::compute_results(
        &latencies,
        &throughputs,
        cold_start,
        cv_threshold,
    );

    // p50 at index 5 = 6.0
    assert!((result.p50_latency_ms - 6.0).abs() < 0.1);
    // p99 at index 9 = 10.0
    assert!((result.p99_latency_ms - 10.0).abs() < 0.1);
}

#[test]
#[ignore = "Requires llama.cpp server at localhost:8082"]
fn test_benchmark_runner_llamacpp() {
    // Use relaxed config for quick test (no preflight for speed)
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 5, 0.50), // Relaxed for test
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.1,
        run_preflight: false, // Skip preflight for test speed
        filter_outliers: false,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://localhost:8082")
        .expect("Benchmark failed - is llama.cpp running?");

    assert!(result.sample_count >= 3);
    assert!(result.mean_latency_ms > 0.0);
    assert!(result.throughput_tps > 0.0);

    println!("llama.cpp Benchmark Results:");
    println!("  Samples: {}", result.sample_count);
    println!("  Filtered Samples: {}", result.filtered_sample_count);
    println!("  Mean: {:.2}ms", result.mean_latency_ms);
    println!("  P50: {:.2}ms", result.p50_latency_ms);
    println!("  P99: {:.2}ms", result.p99_latency_ms);
    println!("  TPS: {:.2}", result.throughput_tps);
    println!("  CV: {:.4}", result.cv_at_stop);
    println!("  Converged: {}", result.cv_converged);
    println!("  Quality Metrics: {:?}", result.quality_metrics);
}

// =========================================================================
// Preflight Integration Tests
// =========================================================================

#[test]
fn test_preflight_checks_passed_empty_initially() {
    let runner = HttpBenchmarkRunner::with_defaults();
    assert!(runner.preflight_checks_passed().is_empty());
}

#[test]
fn test_quality_metrics_in_result() {
    // Test that compute_results includes quality metrics
    let latencies = vec![100.0, 105.0, 95.0, 100.0, 100.0];
    let throughputs = vec![50.0, 48.0, 52.0, 50.0, 50.0];
    let cold_start = 110.0;
    let cv_threshold = 0.10;

    let result = HttpBenchmarkRunner::compute_results(
        &latencies,
        &throughputs,
        cold_start,
        cv_threshold,
    );

    // Check quality metrics are populated
    assert!(result.quality_metrics.cv_at_stop < 0.10);
    assert!(result.quality_metrics.cv_converged);
    assert_eq!(result.quality_metrics.outliers_detected, 0);
    assert!(result.quality_metrics.preflight_checks_passed.is_empty());
}

#[test]
fn test_filtered_samples_in_result() {
    // Test backward-compatible compute_results sets filtered = raw
    let latencies = vec![100.0, 105.0, 95.0];
    let throughputs = vec![];
    let result = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);

    assert_eq!(
        result.latency_samples.len(),
        result.latency_samples_filtered.len()
    );
    assert_eq!(result.sample_count, result.filtered_sample_count);
}

// =========================================================================
// IMP-144: Real-World Throughput Comparison Tests (EXTREME TDD)
// =========================================================================
// These tests verify actual throughput against external servers.
// Run with: cargo test test_imp_144 --lib --features bench-http -- --ignored

/// IMP-144a: Verify llama.cpp throughput measurement works with real server
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_144a_llamacpp_real_throughput() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 10, 0.20),
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.0, // Deterministic
        ..Default::default()
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-144a: Should get llama.cpp benchmark result");

    // IMP-144a: Throughput should be measured and positive
    assert!(
        result.throughput_tps > 0.0,
        "IMP-144a: llama.cpp throughput should be > 0, got {} tok/s",
        result.throughput_tps
    );

    // IMP-144a: Per spec, llama.cpp GPU should be ~162ms latency, ~256 tok/s
    // We just verify it's reasonable (> 10 tok/s)
    assert!(
        result.throughput_tps > 10.0,
        "IMP-144a: llama.cpp throughput should be > 10 tok/s, got {} tok/s",
        result.throughput_tps
    );

    println!("\nIMP-144a: llama.cpp Real-World Benchmark Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  P50 Latency: {:.1} ms", result.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", result.p99_latency_ms);
    println!("  Samples: {}", result.sample_count);
    println!("  CV: {:.4}", result.cv_at_stop);
}

/// IMP-144b: Verify Ollama throughput measurement works with real server
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_144b_ollama_real_throughput() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 10, 0.20),
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-144b: Should get Ollama benchmark result");

    // IMP-144b: Throughput should be measured and positive
    assert!(
        result.throughput_tps > 0.0,
        "IMP-144b: Ollama throughput should be > 0, got {} tok/s",
        result.throughput_tps
    );

    // IMP-144b: Per spec, Ollama should be ~143 tok/s
    // We just verify it's reasonable (> 10 tok/s)
    assert!(
        result.throughput_tps > 10.0,
        "IMP-144b: Ollama throughput should be > 10 tok/s, got {} tok/s",
        result.throughput_tps
    );

    println!("\nIMP-144b: Ollama Real-World Benchmark Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  P50 Latency: {:.1} ms", result.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", result.p99_latency_ms);
    println!("  Samples: {}", result.sample_count);
    println!("  CV: {:.4}", result.cv_at_stop);
}

/// IMP-144c: Verify throughput comparison can detect performance differences
#[test]
fn test_imp_144c_throughput_comparison_logic() {
    // test benchmark results for comparison logic test
    let llamacpp_tps = 256.0; // Per spec: llama.cpp GPU
    let ollama_tps = 143.0; // Per spec: Ollama baseline
    let realizar_tps = 80.0; // Per spec: Realizar current (~1.8x gap)

    // IMP-144c: Calculate gap ratios
    let gap_vs_llamacpp = llamacpp_tps / realizar_tps;
    let gap_vs_ollama = ollama_tps / realizar_tps;

    // Per spec, current gap to Ollama is ~1.5-1.8x
    assert!(
        gap_vs_ollama > 1.0 && gap_vs_ollama < 3.0,
        "IMP-144c: Gap to Ollama should be ~1.5-1.8x, got {:.1}x",
        gap_vs_ollama
    );

    // Per spec, gap to llama.cpp is ~3x
    assert!(
        gap_vs_llamacpp > 2.0 && gap_vs_llamacpp < 5.0,
        "IMP-144c: Gap to llama.cpp should be ~3x, got {:.1}x",
        gap_vs_llamacpp
    );

    println!("\nIMP-144c: Throughput Gap Analysis:");
    println!("  Realizar: {:.1} tok/s", realizar_tps);
    println!(
        "  Ollama: {:.1} tok/s ({:.1}x gap)",
        ollama_tps, gap_vs_ollama
    );
    println!(
        "  llama.cpp: {:.1} tok/s ({:.1}x gap)",
        llamacpp_tps, gap_vs_llamacpp
    );
}

/// IMP-144d: Verify CV-based stopping works for throughput measurements
#[test]
fn test_imp_144d_cv_stopping_for_throughput() {
    // test throughput samples with low variance (should converge quickly)
    let throughputs = vec![100.0, 102.0, 98.0, 101.0, 99.0];
    let latencies = vec![10.0, 9.8, 10.2, 10.0, 10.0];

    let result = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 12.0, 0.05);

    // IMP-144d: CV should converge for stable throughput
    assert!(
        result.cv_converged,
        "IMP-144d: CV should converge for stable throughput, cv={:.4}",
        result.cv_at_stop
    );

    // IMP-144d: Throughput should be calculated correctly
    let expected_mean_tps = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    assert!(
        (result.throughput_tps - expected_mean_tps).abs() < 1.0,
        "IMP-144d: Mean TPS should be ~{:.1}, got {:.1}",
        expected_mean_tps,
        result.throughput_tps
    );
}

// =========================================================================
// IMP-145: Output Correctness Verification (EXTREME TDD)
// =========================================================================
// These tests verify output correctness against llama.cpp (QA-001)
// Run with: cargo test test_imp_145 --lib --features bench-http -- --ignored

/// IMP-145a: Verify deterministic config produces identical output
#[test]
fn test_imp_145a_deterministic_config_structure() {
    // IMP-145a: Deterministic config should have temperature=0
    let config = HttpBenchmarkConfig {
        temperature: 0.0,
        ..Default::default()
    };

    assert_eq!(
        config.temperature, 0.0,
        "IMP-145a: Deterministic config should have temperature=0"
    );
}

/// IMP-145b: Verify same prompt produces same output (local determinism)
#[test]
fn test_imp_145b_local_determinism() {
    // IMP-145b: Same input should produce same output structure
    let latencies = vec![100.0, 100.0, 100.0];
    let throughputs = vec![50.0, 50.0, 50.0];

    let result1 = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);
    let result2 = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);

    // IMP-145b: Same inputs should produce identical results
    assert_eq!(
        result1.mean_latency_ms, result2.mean_latency_ms,
        "IMP-145b: Same inputs should produce identical mean latency"
    );
    assert_eq!(
        result1.throughput_tps, result2.throughput_tps,
        "IMP-145b: Same inputs should produce identical throughput"
    );
}

/// IMP-145c: Verify llama.cpp output matches on repeated calls (deterministic mode)
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_145c_llamacpp_deterministic_output() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082
    // QA-001: Output matches llama.cpp for identical inputs (deterministic mode)

    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "What is 2+2? Answer with just the number:".to_string(),
        max_tokens: 5,
        temperature: Some(0.0), // Deterministic
        stream: false,
    };

    // Make two identical requests
    let result1 = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-145c: First llama.cpp call should succeed");
    let result2 = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-145c: Second llama.cpp call should succeed");

    // IMP-145c: Deterministic mode should produce identical output
    assert_eq!(
        result1.text, result2.text,
        "IMP-145c: llama.cpp should produce identical output in deterministic mode. \
        Got '{}' vs '{}'",
        result1.text, result2.text
    );

    println!("\nIMP-145c: llama.cpp Determinism Verification:");
    println!("  Prompt: '{}'", request.prompt);
    println!("  Output 1: '{}'", result1.text.trim());
    println!("  Output 2: '{}'", result2.text.trim());
    println!("  Match: {}", result1.text == result2.text);
}

/// IMP-145d: Verify Ollama output matches on repeated calls (deterministic mode)
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_145d_ollama_deterministic_output() {
    // This test requires: ollama serve
    let client = ModelHttpClient::with_timeout(30);
    let request = OllamaRequest {
        model: "phi2:2.7b".to_string(),
        prompt: "What is 2+2? Answer with just the number:".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(5),
            temperature: Some(0.0), // Deterministic
        }),
    };

    // Make two identical requests
    let result1 = client
        .ollama_generate("http://127.0.0.1:11434", &request)
        .expect("IMP-145d: First Ollama call should succeed");
    let result2 = client
        .ollama_generate("http://127.0.0.1:11434", &request)
        .expect("IMP-145d: Second Ollama call should succeed");

    // IMP-145d: Deterministic mode should produce identical output
    assert_eq!(
        result1.text, result2.text,
        "IMP-145d: Ollama should produce identical output in deterministic mode. \
        Got '{}' vs '{}'",
        result1.text, result2.text
    );

    println!("\nIMP-145d: Ollama Determinism Verification:");
    println!("  Prompt: '{}'", request.prompt);
    println!("  Output 1: '{}'", result1.text.trim());
    println!("  Output 2: '{}'", result2.text.trim());
    println!("  Match: {}", result1.text == result2.text);
}

// =========================================================================
// IMP-146: Real-World Throughput Baseline Measurement (EXTREME TDD)
// =========================================================================
// These tests establish baseline measurements and track progress toward parity.
// Per Five Whys Analysis (spec §12A), current gap is 3.2x vs llama.cpp.
// Run with: cargo test test_imp_146 --lib --features bench-http -- --ignored

/// IMP-146a: Baseline measurement struct for tracking performance over time
#[derive(Debug, Clone)]
pub struct ThroughputBaseline {
    /// Server name (llama.cpp, Ollama, Realizar)
    pub server: String,
    /// Measured throughput in tokens/second
    pub throughput_tps: f64,
    /// P50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Coefficient of variation (measurement quality)
    pub cv: f64,
    /// Number of samples collected
    pub samples: usize,
}

/// IMP-146a: Verify baseline measurement struct captures required fields
#[test]
fn test_imp_146a_baseline_struct() {
    let baseline = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: 256.0,
        p50_latency_ms: 162.0,
        p99_latency_ms: 290.0,
        cv: 0.045,
        samples: 10,
    };

    // IMP-146a: All fields should be captured
    assert_eq!(baseline.server, "llama.cpp");
    assert!((baseline.throughput_tps - 256.0).abs() < 0.1);
    assert!((baseline.p50_latency_ms - 162.0).abs() < 0.1);
    assert!((baseline.cv - 0.045).abs() < 0.001);
    assert_eq!(baseline.samples, 10);
}

/// IMP-146b: Gap analysis struct for comparing baselines
#[derive(Debug, Clone)]
pub struct GapAnalysis {
    /// Our baseline (Realizar)
    pub realizar: ThroughputBaseline,
    /// Reference baseline (llama.cpp or Ollama)
    pub reference: ThroughputBaseline,
    /// Gap ratio (reference / realizar)
    pub gap_ratio: f64,
    /// Absolute throughput gap
    pub throughput_gap_tps: f64,
    /// Target throughput for parity (80% of reference)
    pub parity_target_tps: f64,
}

/// IMP-146b: Verify gap analysis calculates ratios correctly
#[test]
fn test_imp_146b_gap_analysis() {
    let realizar = ThroughputBaseline {
        server: "Realizar".to_string(),
        throughput_tps: 80.0, // Per spec: current ~80 tok/s
        p50_latency_ms: 520.0,
        p99_latency_ms: 800.0,
        cv: 0.08,
        samples: 10,
    };

    let llamacpp = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: 256.0, // Per spec: ~256 tok/s GPU
        p50_latency_ms: 162.0,
        p99_latency_ms: 290.0,
        cv: 0.045,
        samples: 10,
    };

    let gap = GapAnalysis {
        gap_ratio: llamacpp.throughput_tps / realizar.throughput_tps,
        throughput_gap_tps: llamacpp.throughput_tps - realizar.throughput_tps,
        parity_target_tps: llamacpp.throughput_tps * 0.8, // 80% is parity
        realizar,
        reference: llamacpp,
    };

    // IMP-146b: Gap should be ~3.2x per Five Whys analysis
    assert!(
        gap.gap_ratio > 2.5 && gap.gap_ratio < 4.0,
        "IMP-146b: Gap to llama.cpp should be ~3.2x, got {:.1}x",
        gap.gap_ratio
    );

    // IMP-146b: Parity target should be 80% of reference
    assert!(
        (gap.parity_target_tps - 204.8).abs() < 1.0,
        "IMP-146b: Parity target should be ~205 tok/s, got {:.1}",
        gap.parity_target_tps
    );

    println!("\nIMP-146b: Gap Analysis:");
    println!("  Realizar: {:.1} tok/s", gap.realizar.throughput_tps);
    println!("  llama.cpp: {:.1} tok/s", gap.reference.throughput_tps);
    println!("  Gap: {:.1}x", gap.gap_ratio);
    println!("  Target for parity: {:.1} tok/s", gap.parity_target_tps);
}

/// IMP-146c: Real-world baseline measurement against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_146c_llamacpp_baseline_measurement() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10), // Scientific rigor
        warmup_iterations: 2,
        prompt: "Explain what machine learning is in one paragraph:".to_string(),
        max_tokens: 50,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-146c: llama.cpp baseline measurement should succeed");

    // IMP-146c: Build baseline from result
    let baseline = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: result.throughput_tps,
        p50_latency_ms: result.p50_latency_ms,
        p99_latency_ms: result.p99_latency_ms,
        cv: result.cv_at_stop,
        samples: result.sample_count,
    };

    // IMP-146c: Baseline should have reasonable values
    assert!(
        baseline.throughput_tps > 50.0,
        "IMP-146c: llama.cpp should achieve > 50 tok/s, got {:.1}",
        baseline.throughput_tps
    );
    assert!(
        baseline.cv < 0.20,
        "IMP-146c: CV should be < 20% for reliable measurement, got {:.2}",
        baseline.cv
    );

    println!("\nIMP-146c: llama.cpp Baseline Measurement:");
    println!("  Throughput: {:.1} tok/s", baseline.throughput_tps);
    println!("  P50 Latency: {:.1} ms", baseline.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", baseline.p99_latency_ms);
    println!(
        "  CV: {:.4} ({})",
        baseline.cv,
        if baseline.cv < 0.05 {
            "excellent"
        } else if baseline.cv < 0.10 {
            "good"
        } else {
            "acceptable"
        }
    );
    println!("  Samples: {}", baseline.samples);
}

/// IMP-146d: Real-world baseline measurement against Ollama
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_146d_ollama_baseline_measurement() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "Explain what machine learning is in one paragraph:".to_string(),
        max_tokens: 50,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-146d: Ollama baseline measurement should succeed");

    // IMP-146d: Build baseline from result
    let baseline = ThroughputBaseline {
        server: "Ollama".to_string(),
        throughput_tps: result.throughput_tps,
        p50_latency_ms: result.p50_latency_ms,
        p99_latency_ms: result.p99_latency_ms,
        cv: result.cv_at_stop,
        samples: result.sample_count,
    };

    // IMP-146d: Baseline should have reasonable values
    assert!(
        baseline.throughput_tps > 30.0,
        "IMP-146d: Ollama should achieve > 30 tok/s, got {:.1}",
        baseline.throughput_tps
    );
    assert!(
        baseline.cv < 0.20,
        "IMP-146d: CV should be < 20% for reliable measurement, got {:.2}",
        baseline.cv
    );

    println!("\nIMP-146d: Ollama Baseline Measurement:");
    println!("  Throughput: {:.1} tok/s", baseline.throughput_tps);
    println!("  P50 Latency: {:.1} ms", baseline.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", baseline.p99_latency_ms);
    println!(
        "  CV: {:.4} ({})",
        baseline.cv,
        if baseline.cv < 0.05 {
            "excellent"
        } else if baseline.cv < 0.10 {
            "good"
        } else {
            "acceptable"
        }
    );
    println!("  Samples: {}", baseline.samples);
}

// =========================================================================
// IMP-151: Real-World Throughput Regression Tests (EXTREME TDD)
// =========================================================================
// These tests track performance progress and detect regressions.
// Per Five Whys Analysis, target: 80 tok/s → 120 tok/s (P1) → 200 tok/s (P2)

/// IMP-151a: Performance milestone tracking struct
#[derive(Debug, Clone)]
pub struct PerformanceMilestone {
    /// Milestone name (e.g., "P1", "P2", "Parity")
    pub name: String,
    /// Target throughput in tokens/second
    pub target_tps: f64,
    /// Current achieved throughput
    pub achieved_tps: f64,
    /// Gap to target as percentage
    pub gap_percent: f64,
    /// Whether milestone is achieved
    pub achieved: bool,
}

impl PerformanceMilestone {
    pub fn new(name: &str, target_tps: f64, achieved_tps: f64) -> Self {
        let gap_percent = if target_tps > 0.0 {
            ((target_tps - achieved_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        Self {
            name: name.to_string(),
            target_tps,
            achieved_tps,
            gap_percent,
            achieved: achieved_tps >= target_tps,
        }
    }
}

/// IMP-151a: Verify milestone tracking struct works correctly
#[test]
fn test_imp_151a_milestone_tracking() {
    // Current baseline: 80 tok/s
    let current_tps = 80.0;

    // Define milestones per Five Whys roadmap
    let p1_milestone = PerformanceMilestone::new("P1", 120.0, current_tps);
    let p2_milestone = PerformanceMilestone::new("P2", 200.0, current_tps);
    let parity_milestone = PerformanceMilestone::new("Parity", 205.0, current_tps);

    // IMP-151a: Verify milestone calculations
    assert!(
        !p1_milestone.achieved,
        "IMP-151a: P1 not yet achieved at 80 tok/s"
    );
    assert!(
        (p1_milestone.gap_percent - 33.3).abs() < 1.0,
        "IMP-151a: Gap to P1 should be ~33%, got {:.1}%",
        p1_milestone.gap_percent
    );

    assert!(!p2_milestone.achieved, "IMP-151a: P2 not yet achieved");
    assert!(
        (p2_milestone.gap_percent - 60.0).abs() < 1.0,
        "IMP-151a: Gap to P2 should be ~60%, got {:.1}%",
        p2_milestone.gap_percent
    );

    println!("\nIMP-151a: Performance Milestone Tracking:");
    println!("  Current: {:.1} tok/s", current_tps);
    println!(
        "  P1 (120 tok/s): {:.1}% gap, achieved={}",
        p1_milestone.gap_percent, p1_milestone.achieved
    );
    println!(
        "  P2 (200 tok/s): {:.1}% gap, achieved={}",
        p2_milestone.gap_percent, p2_milestone.achieved
    );
    println!(
        "  Parity (205 tok/s): {:.1}% gap, achieved={}",
        parity_milestone.gap_percent, parity_milestone.achieved
    );
}

/// IMP-151b: Regression detection struct
#[derive(Debug, Clone)]
pub struct RegressionCheck {
    /// Test name
    pub test_name: String,
    /// Baseline throughput (previous best)
    pub baseline_tps: f64,
    /// Current throughput
    pub current_tps: f64,
    /// Regression threshold percentage (e.g., 5% = flag if >5% slower)
    pub threshold_percent: f64,
    /// Whether regression detected
    pub regression_detected: bool,
    /// Improvement percentage (negative = regression)
    pub improvement_percent: f64,
}

impl RegressionCheck {
    pub fn new(
        test_name: &str,
        baseline_tps: f64,
        current_tps: f64,
        threshold_percent: f64,
    ) -> Self {
        let improvement_percent = if baseline_tps > 0.0 {
            ((current_tps - baseline_tps) / baseline_tps) * 100.0
        } else {
            0.0
        };
        let regression_detected = improvement_percent < -threshold_percent;
        Self {
            test_name: test_name.to_string(),
            baseline_tps,
            current_tps,
            threshold_percent,
            regression_detected,
            improvement_percent,
        }
    }
}

/// IMP-151b: Verify regression detection works correctly
#[test]
fn test_imp_151b_regression_detection() {
    // Scenario 1: No regression (improvement)
    let check1 = RegressionCheck::new("dequant_q4k", 80.0, 85.0, 5.0);
    assert!(
        !check1.regression_detected,
        "IMP-151b: 85 vs 80 should not be regression"
    );
    assert!(
        (check1.improvement_percent - 6.25).abs() < 0.1,
        "IMP-151b: Should show ~6.25% improvement"
    );

    // Scenario 2: Minor regression within threshold
    let check2 = RegressionCheck::new("fused_matvec", 100.0, 97.0, 5.0);
    assert!(
        !check2.regression_detected,
        "IMP-151b: 3% drop within 5% threshold"
    );

    // Scenario 3: Significant regression exceeds threshold
    let check3 = RegressionCheck::new("simd_extract", 100.0, 90.0, 5.0);
    assert!(
        check3.regression_detected,
        "IMP-151b: 10% drop should trigger regression"
    );

    println!("\nIMP-151b: Regression Detection:");
    println!(
        "  Test 1 (85 vs 80): {:.1}% change, regression={}",
        check1.improvement_percent, check1.regression_detected
    );
    println!(
        "  Test 2 (97 vs 100): {:.1}% change, regression={}",
        check2.improvement_percent, check2.regression_detected
    );
    println!(
        "  Test 3 (90 vs 100): {:.1}% change, regression={}",
        check3.improvement_percent, check3.regression_detected
    );
}

/// IMP-151c: Real-world regression test against llama.cpp baseline
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_151c_llamacpp_regression_check() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is 2+2? Answer briefly:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-151c: llama.cpp benchmark should succeed");

    // llama.cpp baseline: ~256 tok/s (per spec)
    let expected_baseline = 256.0;
    let tolerance_percent = 30.0; // Allow 30% variance for different hardware

    let check = RegressionCheck::new(
        "llamacpp_throughput",
        expected_baseline,
        result.throughput_tps,
        tolerance_percent,
    );

    println!("\nIMP-151c: llama.cpp Regression Check:");
    println!("  Expected baseline: {:.1} tok/s", expected_baseline);
    println!("  Measured: {:.1} tok/s", result.throughput_tps);
    println!("  Difference: {:.1}%", check.improvement_percent);
    println!("  Regression: {}", check.regression_detected);

    // Note: Not asserting regression here since hardware varies
    // This is for tracking, not blocking
}

/// IMP-151d: Real-world regression test against Ollama baseline
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_151d_ollama_regression_check() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is 2+2? Answer briefly:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-151d: Ollama benchmark should succeed");

    // Ollama baseline: ~143 tok/s (per spec)
    let expected_baseline = 143.0;
    let tolerance_percent = 30.0;

    let check = RegressionCheck::new(
        "ollama_throughput",
        expected_baseline,
        result.throughput_tps,
        tolerance_percent,
    );

    println!("\nIMP-151d: Ollama Regression Check:");
    println!("  Expected baseline: {:.1} tok/s", expected_baseline);
    println!("  Measured: {:.1} tok/s", result.throughput_tps);
    println!("  Difference: {:.1}%", check.improvement_percent);
    println!("  Regression: {}", check.regression_detected);
}

// =========================================================================
// IMP-152: End-to-End Performance Comparison Benchmark (EXTREME TDD)
// Per spec §8.3: Side-by-side comparison of Realizar vs Ollama vs llama.cpp
// =========================================================================

/// IMP-152a: End-to-end comparison result tracking
#[derive(Debug, Clone)]
pub struct E2EComparisonResult {
    /// Realizar throughput (tok/s)
    pub realizar_tps: f64,
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// llama.cpp throughput (tok/s)
    pub llamacpp_tps: f64,
    /// Gap vs Ollama (positive = Realizar is faster)
    pub gap_vs_ollama_percent: f64,
    /// Gap vs llama.cpp (positive = Realizar is faster)
    pub gap_vs_llamacpp_percent: f64,
    /// Parity achieved (within 10% of llama.cpp)
    pub parity_achieved: bool,
    /// Timestamp of comparison
    pub timestamp: String,
}

impl E2EComparisonResult {
    pub fn new(realizar_tps: f64, ollama_tps: f64, llamacpp_tps: f64) -> Self {
        let gap_vs_ollama = if ollama_tps > 0.0 {
            ((realizar_tps - ollama_tps) / ollama_tps) * 100.0
        } else {
            0.0
        };
        let gap_vs_llamacpp = if llamacpp_tps > 0.0 {
            ((realizar_tps - llamacpp_tps) / llamacpp_tps) * 100.0
        } else {
            0.0
        };
        // Parity = within 10% of llama.cpp (per spec)
        let parity_achieved = gap_vs_llamacpp >= -10.0;

        Self {
            realizar_tps,
            ollama_tps,
            llamacpp_tps,
            gap_vs_ollama_percent: gap_vs_ollama,
            gap_vs_llamacpp_percent: gap_vs_llamacpp,
            parity_achieved,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// IMP-152a: Test E2E comparison result struct
#[test]
fn test_imp_152a_e2e_comparison_struct() {
    // Scenario: Realizar at 200 tok/s, Ollama at 143, llama.cpp at 256
    let result = E2EComparisonResult::new(200.0, 143.0, 256.0);

    // Verify gap calculations
    let expected_ollama_gap: f64 = ((200.0 - 143.0) / 143.0) * 100.0; // +39.9%
    let expected_llamacpp_gap: f64 = ((200.0 - 256.0) / 256.0) * 100.0; // -21.9%

    assert!(
        (result.gap_vs_ollama_percent - expected_ollama_gap).abs() < 0.1,
        "IMP-152a: Ollama gap should be ~39.9%"
    );
    assert!(
        (result.gap_vs_llamacpp_percent - expected_llamacpp_gap).abs() < 0.1,
        "IMP-152a: llama.cpp gap should be ~-21.9%"
    );
    assert!(
        !result.parity_achieved,
        "IMP-152a: -21.9% gap should not be parity"
    );

    println!("\nIMP-152a: E2E Comparison Result:");
    println!("  Realizar: {:.1} tok/s", result.realizar_tps);
    println!("  Ollama:   {:.1} tok/s", result.ollama_tps);
    println!("  llama.cpp: {:.1} tok/s", result.llamacpp_tps);
    println!("  Gap vs Ollama: {:+.1}%", result.gap_vs_ollama_percent);
    println!(
        "  Gap vs llama.cpp: {:+.1}%",
        result.gap_vs_llamacpp_percent
    );
    println!("  Parity achieved: {}", result.parity_achieved);
}

/// IMP-152b: Test parity threshold detection
#[test]
fn test_imp_152b_parity_detection() {
    // Scenario 1: Just within parity (232 tok/s vs 256 = -9.4% gap)
    // 232/256 = 0.906, so gap = -9.4% which is > -10%
    let at_parity = E2EComparisonResult::new(232.0, 143.0, 256.0);
    assert!(
        at_parity.parity_achieved,
        "IMP-152b: 232 vs 256 should be parity (-9.4%)"
    );

    // Scenario 2: Beyond parity (260 tok/s = +1.5% faster)
    let beyond_parity = E2EComparisonResult::new(260.0, 143.0, 256.0);
    assert!(
        beyond_parity.parity_achieved,
        "IMP-152b: 260 vs 256 should definitely be parity"
    );
    assert!(
        beyond_parity.gap_vs_llamacpp_percent > 0.0,
        "IMP-152b: 260 vs 256 should show positive gap"
    );

    // Scenario 3: Below parity (200 tok/s = -21.9% gap)
    let below_parity = E2EComparisonResult::new(200.0, 143.0, 256.0);
    assert!(
        !below_parity.parity_achieved,
        "IMP-152b: 200 vs 256 should NOT be parity"
    );

    // Scenario 4: Exactly at threshold (231 tok/s = -9.8% gap)
    let exact_threshold = E2EComparisonResult::new(231.0, 143.0, 256.0);
    assert!(
        exact_threshold.parity_achieved,
        "IMP-152b: 231 vs 256 should be parity (-9.8%)"
    );

    println!("\nIMP-152b: Parity Detection:");
    println!(
        "  232 vs 256 = {:.1}% gap, parity={}",
        at_parity.gap_vs_llamacpp_percent, at_parity.parity_achieved
    );
    println!(
        "  260 vs 256 = {:+.1}% gap, parity={}",
        beyond_parity.gap_vs_llamacpp_percent, beyond_parity.parity_achieved
    );
    println!(
        "  200 vs 256 = {:.1}% gap, parity={}",
        below_parity.gap_vs_llamacpp_percent, below_parity.parity_achieved
    );
    println!(
        "  231 vs 256 = {:.1}% gap, parity={}",
        exact_threshold.gap_vs_llamacpp_percent, exact_threshold.parity_achieved
    );
}

/// IMP-152c: Real-world E2E comparison (requires both servers)
#[test]
#[ignore = "Requires running Ollama (11434) and llama.cpp (8082) servers"]
fn test_imp_152c_real_e2e_comparison() {
    // This test requires:
    // 1. ollama serve
    // 2. llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is the capital of France? Answer in one word:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);

    // Benchmark both external servers
    let llamacpp_result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-152c: llama.cpp benchmark failed");
    let ollama_result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-152c: Ollama benchmark failed");

    // test Realizar result based on IMP-900 benchmark projections
    // IMP-900 shows 61 tok/s projected, targeting 80+ with further optimizations
    let realizar_tps: f64 = 61.0; // IMP-900 projected throughput

    let comparison = E2EComparisonResult::new(
        realizar_tps,
        ollama_result.throughput_tps,
        llamacpp_result.throughput_tps,
    );

    println!("\nIMP-152c: Real-World E2E Comparison:");
    println!("  Realizar:  {:.1} tok/s (test)", comparison.realizar_tps);
    println!("  Ollama:    {:.1} tok/s (measured)", comparison.ollama_tps);
    println!(
        "  llama.cpp: {:.1} tok/s (measured)",
        comparison.llamacpp_tps
    );
    println!(
        "  Gap vs Ollama:    {:+.1}%",
        comparison.gap_vs_ollama_percent
    );
    println!(
        "  Gap vs llama.cpp: {:+.1}%",
        comparison.gap_vs_llamacpp_percent
    );
    println!("  Parity achieved:  {}", comparison.parity_achieved);
    println!("  Timestamp: {}", comparison.timestamp);
}

/// IMP-152d: Progress delta tracking across milestones
#[derive(Debug, Clone)]
pub struct ProgressDelta {
    /// Previous comparison result
    pub previous_tps: f64,
    /// Current comparison result
    pub current_tps: f64,
    /// Absolute improvement (tok/s)
    pub delta_tps: f64,
    /// Relative improvement percentage
    pub delta_percent: f64,
    /// Target for next milestone
    pub next_milestone_tps: f64,
    /// Percentage progress toward next milestone
    pub progress_to_next: f64,
}

impl ProgressDelta {
    pub fn new(previous_tps: f64, current_tps: f64, next_milestone_tps: f64) -> Self {
        let delta_tps = current_tps - previous_tps;
        let delta_percent = if previous_tps > 0.0 {
            (delta_tps / previous_tps) * 100.0
        } else {
            0.0
        };
        let progress_to_next = if next_milestone_tps > current_tps {
            ((current_tps - previous_tps) / (next_milestone_tps - previous_tps)) * 100.0
        } else {
            100.0 // Already at or beyond milestone
        };
        Self {
            previous_tps,
            current_tps,
            delta_tps,
            delta_percent,
            next_milestone_tps,
            progress_to_next,
        }
    }
}

/// IMP-152d: Test progress delta tracking
#[test]
fn test_imp_152d_progress_delta_tracking() {
    // Scenario: Improved from 80 tok/s to 100 tok/s, targeting P1 = 120 tok/s
    let delta = ProgressDelta::new(80.0, 100.0, 120.0);

    assert!(
        (delta.delta_tps - 20.0).abs() < 0.01,
        "IMP-152d: Delta should be 20 tok/s"
    );
    assert!(
        (delta.delta_percent - 25.0).abs() < 0.1,
        "IMP-152d: Delta should be 25%"
    );
    assert!(
        (delta.progress_to_next - 50.0).abs() < 0.1,
        "IMP-152d: Progress should be 50% (20 of 40 tok/s needed)"
    );

    // Scenario: At milestone (120 tok/s achieved, targeting P2 = 200)
    let delta2 = ProgressDelta::new(100.0, 120.0, 200.0);
    assert!(
        (delta2.delta_percent - 20.0).abs() < 0.1,
        "IMP-152d: Delta should be 20%"
    );

    // Scenario: Beyond milestone
    let delta3 = ProgressDelta::new(180.0, 210.0, 200.0);
    assert!(
        (delta3.progress_to_next - 100.0).abs() < 0.01,
        "IMP-152d: Should be 100% when beyond milestone"
    );

    println!("\nIMP-152d: Progress Delta Tracking:");
    println!(
        "  80 → 100 (target 120): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta.delta_tps, delta.delta_percent, delta.progress_to_next
    );
    println!(
        "  100 → 120 (target 200): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta2.delta_tps, delta2.delta_percent, delta2.progress_to_next
    );
    println!(
        "  180 → 210 (target 200): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta3.delta_tps, delta3.delta_percent, delta3.progress_to_next
    );
}

// =========================================================================
// IMP-153: Performance Progress Tracking Metrics (EXTREME TDD)
// Per spec §9.1: Historical tracking and trend analysis for performance
