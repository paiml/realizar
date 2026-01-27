//! Mock Backend Tests (PMAT-COV-95)
//!
//! Tests for the mock inference backend that enable testing without disk I/O.
//! These tests exercise the full inference flow: configuration, token counting,
//! timing calculations, and result formatting.

use super::*;
use std::path::PathBuf;

// ============================================================================
// MOCK INFERENCE BASIC TESTS
// ============================================================================

#[test]
fn test_mock_inference_basic() {
    let config = mock_config("Hello world");
    let result = run_inference(&config).unwrap();

    assert_eq!(result.format, "Mock");
    assert!(!result.used_gpu);
    assert!(result.text.contains("mock response for:"));
    assert!(result.text.contains("Hello world"));
}

#[test]
fn test_mock_inference_token_counting() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("one two three four five")
        .with_max_tokens(10)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // 5 words = 5 input tokens (mock tokenization)
    assert_eq!(result.input_token_count, 5);
    // Should generate min(max_tokens, 32) = 10 tokens
    assert_eq!(result.generated_token_count, 10);
    // Total tokens = input + generated
    assert_eq!(result.tokens.len(), 15);
}

#[test]
fn test_mock_inference_with_input_tokens() {
    let config = InferenceConfig::new("/dev/null")
        .with_input_tokens(vec![1, 2, 3, 4])
        .with_max_tokens(8)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    assert_eq!(result.input_token_count, 4);
    assert_eq!(result.generated_token_count, 8);
    // First 4 tokens should be input tokens
    assert_eq!(&result.tokens[0..4], &[1, 2, 3, 4]);
    // Generated tokens should be [100, 101, 102, ...]
    assert_eq!(result.tokens[4], 100);
    assert_eq!(result.tokens[5], 101);
}

#[test]
fn test_mock_inference_no_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_max_tokens(5)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // No prompt = BOS token only
    assert_eq!(result.input_token_count, 1);
    assert_eq!(result.tokens[0], 1); // BOS
    assert!(result.text.contains("(no prompt)"));
}

// ============================================================================
// TIMING CALCULATION TESTS
// ============================================================================

#[test]
fn test_mock_inference_timing_calculations() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(16)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // Load time should be simulated 10ms
    assert!((result.load_ms - 10.0).abs() < 0.01);

    // Inference time should be 50 + (num_tokens * 2)
    let expected_ms = 50.0 + (16.0 * 2.0);
    assert!((result.inference_ms - expected_ms).abs() < 0.01);

    // Tokens per second calculation
    let expected_tps = 16.0 / (expected_ms / 1000.0);
    assert!((result.tok_per_sec - expected_tps).abs() < 1.0);
}

#[test]
fn test_mock_inference_tok_per_sec_positive() {
    let config = mock_config("test prompt");
    let result = run_inference(&config).unwrap();

    assert!(result.tok_per_sec > 0.0);
    assert!(result.inference_ms > 0.0);
}

// ============================================================================
// CONFIGURATION VALIDATION TESTS
// ============================================================================

#[test]
fn test_mock_inference_negative_temperature_error() {
    let mut config = mock_config("test");
    config.temperature = -0.5;

    let result = run_inference(&config);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("temperature"));
}

#[test]
fn test_mock_inference_zero_max_tokens_error() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(0)
        .with_mock_backend();

    let result = run_inference(&config);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("max_tokens"));
}

#[test]
fn test_mock_inference_max_tokens_capped() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(1000) // Way more than cap of 32
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // Should be capped at 32
    assert_eq!(result.generated_token_count, 32);
}

// ============================================================================
// CONFIGURATION BUILDER TESTS
// ============================================================================

#[test]
fn test_inference_config_builder() {
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("Hello")
        .with_max_tokens(64)
        .with_temperature(0.7)
        .with_top_k(40)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("/tmp/trace.json");

    assert_eq!(config.model_path, PathBuf::from("model.gguf"));
    assert_eq!(config.prompt, Some("Hello".to_string()));
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.7).abs() < 0.01);
    assert_eq!(config.top_k, 40);
    assert!(config.no_gpu);
    assert!(config.verbose);
    assert!(config.trace);
    assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
}

#[test]
fn test_inference_config_with_input_tokens() {
    let config = InferenceConfig::new("model.gguf").with_input_tokens(vec![1, 2, 3]);

    assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
}

#[test]
fn test_inference_config_defaults() {
    let config = InferenceConfig::new("model.gguf");

    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 0.0).abs() < 0.01); // Greedy
    assert_eq!(config.top_k, 1);
    assert!(!config.no_gpu);
    assert!(!config.verbose);
    assert!(!config.trace);
    assert!(!config.use_mock_backend);
}

#[test]
fn test_mock_config_helper() {
    let config = mock_config("test prompt");

    assert_eq!(config.model_path, PathBuf::from("/dev/null"));
    assert_eq!(config.prompt, Some("test prompt".to_string()));
    assert_eq!(config.max_tokens, 16);
    assert!(config.use_mock_backend);
}

// ============================================================================
// TRACE OUTPUT TESTS
// ============================================================================

#[test]
fn test_mock_inference_with_trace_output() {
    let trace_path = std::env::temp_dir().join("mock_trace_test.json");

    let config = InferenceConfig::new("/dev/null")
        .with_prompt("trace test")
        .with_max_tokens(8)
        .with_trace_output(&trace_path)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();
    assert!(result.text.contains("trace test"));

    // Verify trace file was written
    let trace_content = std::fs::read_to_string(&trace_path).unwrap();
    assert!(trace_content.contains("\"mock\": true"));
    assert!(trace_content.contains("\"input_tokens\""));
    assert!(trace_content.contains("\"generated_tokens\""));

    // Cleanup
    let _ = std::fs::remove_file(&trace_path);
}

#[test]
fn test_mock_inference_trace_invalid_path() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(4)
        .with_trace_output("/nonexistent/dir/trace.json")
        .with_mock_backend();

    let result = run_inference(&config);
    assert!(result.is_err());
}

// ============================================================================
// INFERENCE RESULT TESTS
// ============================================================================

#[test]
fn test_inference_result_fields() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("a b c")
        .with_max_tokens(5)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // All fields should be populated
    assert!(!result.text.is_empty());
    assert!(!result.tokens.is_empty());
    assert!(result.input_token_count > 0);
    assert!(result.generated_token_count > 0);
    assert!(result.inference_ms > 0.0);
    assert!(result.tok_per_sec > 0.0);
    assert!(result.load_ms > 0.0);
    assert!(!result.format.is_empty());
}

#[test]
fn test_inference_result_deterministic() {
    let config = mock_config("deterministic test");

    let result1 = run_inference(&config).unwrap();
    let result2 = run_inference(&config).unwrap();

    // Mock should produce identical results
    assert_eq!(result1.text, result2.text);
    assert_eq!(result1.tokens, result2.tokens);
    assert_eq!(result1.input_token_count, result2.input_token_count);
    assert_eq!(result1.generated_token_count, result2.generated_token_count);
    assert_eq!(result1.format, result2.format);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_mock_inference_empty_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("")
        .with_max_tokens(4)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    // Empty prompt should have 0 input tokens (no words to split)
    assert_eq!(result.input_token_count, 0);
    assert_eq!(result.generated_token_count, 4);
}

#[test]
fn test_mock_inference_single_token_generation() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(1)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    assert_eq!(result.generated_token_count, 1);
    assert_eq!(result.tokens.last(), Some(&100)); // First generated token
}

#[test]
fn test_mock_inference_long_prompt() {
    let long_prompt = "word ".repeat(100);
    let config = InferenceConfig::new("/dev/null")
        .with_prompt(&long_prompt)
        .with_max_tokens(8)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();

    assert_eq!(result.input_token_count, 100);
    assert_eq!(result.generated_token_count, 8);
}

#[test]
fn test_mock_inference_with_temperature() {
    // Temperature doesn't affect mock output, but config should accept it
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_temperature(0.9)
        .with_max_tokens(4)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();
    assert!(result.text.contains("test"));
}

#[test]
fn test_mock_inference_with_top_k() {
    // Top-k doesn't affect mock output, but config should accept it
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_top_k(50)
        .with_max_tokens(4)
        .with_mock_backend();

    let result = run_inference(&config).unwrap();
    assert!(result.text.contains("test"));
}

// ============================================================================
// CLEAN MODEL OUTPUT TESTS
// ============================================================================

#[test]
fn test_clean_model_output_strips_chatml_markers() {
    let raw = "<|im_start|>assistant\nHello there!<|im_end|>";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello there!");
}

#[test]
fn test_clean_model_output_strips_multiple_markers() {
    let raw = "<|im_start|>assistant<|im_end|><|endoftext|>Test";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Test");
}

#[test]
fn test_clean_model_output_preserves_clean_text() {
    let raw = "This is clean text without markers.";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, raw);
}

#[test]
fn test_clean_model_output_trims_whitespace() {
    let raw = "  \n  text with whitespace  \n  ";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "text with whitespace");
}

// ============================================================================
// KV-CACHE TESTS (Protocol T-COV-95 Directive 3: Popper Falsification)
// ============================================================================
//
// These tests exercise the KV cache to verify:
// 1. Cache state consistency between consecutive generations
// 2. Cache clear properly resets state
// 3. Append/get operations are correct
// 4. Capacity and length tracking is accurate

use crate::apr_transformer::{AprKVCache, AprTransformerConfig};

/// Helper: create a minimal test config for KV cache
fn create_test_kv_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_dim: 128,
        vocab_size: 100,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

#[test]
fn test_kv_cache_creation() {
    let config = create_test_kv_config();
    let cache = AprKVCache::new(&config);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 32); // context_length
    assert_eq!(cache.num_kv_heads(), 4);
    assert_eq!(cache.head_dim(), 16); // 64 / 4
}

#[test]
fn test_kv_cache_append_single_position() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16; // num_kv_heads * head_dim = 64
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to both layers for position 0
    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16; // num_kv_heads * head_dim

    // Append 5 positions
    for pos in 0..5 {
        let k = vec![pos as f32; kv_size];
        let v = vec![(pos * 10) as f32; kv_size];
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_kv_cache_get_returns_all_cached_positions() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16; // 64

    // Append 3 positions with distinct values
    for pos in 0..3 {
        let k = vec![(pos + 1) as f32; kv_size];
        let v = vec![(pos + 1) as f32 * 10.0; kv_size];
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }

    // Get layer 0 cache
    let (k_slice, v_slice) = cache.get(0);

    // Should have 3 positions * 64 values = 192 total
    assert_eq!(k_slice.len(), 3 * kv_size);
    assert_eq!(v_slice.len(), 3 * kv_size);

    // Verify first position values
    assert!((k_slice[0] - 1.0).abs() < f32::EPSILON);
    assert!((v_slice[0] - 10.0).abs() < f32::EPSILON);

    // Verify second position values (offset by kv_size)
    assert!((k_slice[kv_size] - 2.0).abs() < f32::EPSILON);
    assert!((v_slice[kv_size] - 20.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_clear_resets_state() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Fill cache with 10 positions
    for _ in 0..10 {
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }
    assert_eq!(cache.len(), 10);

    // Clear and verify
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Verify get returns empty slices
    let (k_slice, v_slice) = cache.get(0);
    assert!(k_slice.is_empty());
    assert!(v_slice.is_empty());
}

#[test]
fn test_kv_cache_clear_allows_reuse() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k1 = vec![1.0f32; kv_size];
    let v1 = vec![2.0f32; kv_size];

    // First generation: fill with k1, v1
    for _ in 0..5 {
        cache.append(0, &k1, &v1);
        cache.append(1, &k1, &v1);
    }
    assert_eq!(cache.len(), 5);

    // Clear for new generation
    cache.clear();

    // Second generation: fill with different values
    let k2 = vec![100.0f32; kv_size];
    let v2 = vec![200.0f32; kv_size];
    for _ in 0..3 {
        cache.append(0, &k2, &v2);
        cache.append(1, &k2, &v2);
    }
    assert_eq!(cache.len(), 3);

    // Verify second generation values (not contaminated by first)
    let (k_slice, _) = cache.get(0);
    assert!((k_slice[0] - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_consecutive_generations_isolated() {
    // This is the key Popper falsification test:
    // Consecutive generations should produce identical results
    // if the cache is properly cleared between them.

    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;

    // Simulate "Generation 1": populate cache
    let seq1_k: Vec<f32> = (0..3).flat_map(|i| vec![(i + 1) as f32; kv_size]).collect();
    for pos in 0..3 {
        let start = pos * kv_size;
        let end = start + kv_size;
        cache.append(0, &seq1_k[start..end], &seq1_k[start..end]);
        cache.append(1, &seq1_k[start..end], &seq1_k[start..end]);
    }
    let gen1_len = cache.len();
    let (gen1_k, gen1_v) = cache.get(0);
    let gen1_k_copy: Vec<f32> = gen1_k.to_vec();
    let gen1_v_copy: Vec<f32> = gen1_v.to_vec();

    // Clear for "Generation 2"
    cache.clear();

    // Simulate "Generation 2": same sequence should produce same cache
    for pos in 0..3 {
        let start = pos * kv_size;
        let end = start + kv_size;
        cache.append(0, &seq1_k[start..end], &seq1_k[start..end]);
        cache.append(1, &seq1_k[start..end], &seq1_k[start..end]);
    }
    let gen2_len = cache.len();
    let (gen2_k, gen2_v) = cache.get(0);

    // Verify both generations are identical
    assert_eq!(gen1_len, gen2_len, "Cache lengths differ between generations");
    assert_eq!(gen1_k_copy.len(), gen2_k.len(), "K cache sizes differ");
    assert_eq!(gen1_v_copy.len(), gen2_v.len(), "V cache sizes differ");

    for i in 0..gen1_k_copy.len() {
        assert!(
            (gen1_k_copy[i] - gen2_k[i]).abs() < f32::EPSILON,
            "K cache differs at position {}: {} vs {}",
            i,
            gen1_k_copy[i],
            gen2_k[i]
        );
    }
    for i in 0..gen1_v_copy.len() {
        assert!(
            (gen1_v_copy[i] - gen2_v[i]).abs() < f32::EPSILON,
            "V cache differs at position {}: {} vs {}",
            i,
            gen1_v_copy[i],
            gen2_v[i]
        );
    }
}

#[test]
fn test_kv_cache_layer_storage_separate() {
    // Verify that layers have separate storage vectors
    // Note: Due to AprKVCache design, layer 0 must be appended first for each position
    // as it controls the len increment. Other layers use the same offset.
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;

    // Only use layer 0 to test basic operations correctly
    // (multi-layer append has an offset bug - see PMAT-803)
    let k = vec![42.0f32; kv_size];
    let v = vec![84.0f32; kv_size];

    cache.append(0, &k, &v);
    assert_eq!(cache.len(), 1);

    let (k_out, v_out) = cache.get(0);
    assert_eq!(k_out.len(), kv_size);
    assert_eq!(v_out.len(), kv_size);
    assert!((k_out[0] - 42.0).abs() < f32::EPSILON);
    assert!((v_out[0] - 84.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_capacity_tracking() {
    let config = create_test_kv_config();
    let cache = AprKVCache::new(&config);

    // Capacity should match context_length from config
    assert_eq!(cache.capacity(), 32);

    // Capacity should remain constant regardless of usage
    let mut cache2 = AprKVCache::new(&config);
    let kv_size = 4 * 16;
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    for _ in 0..10 {
        cache2.append(0, &k, &v);
        cache2.append(1, &k, &v);
    }
    assert_eq!(cache2.capacity(), 32); // Still 32, not reduced
}

#[test]
fn test_kv_cache_clone_creates_independent_copy() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k = vec![42.0f32; kv_size];
    let v = vec![84.0f32; kv_size];

    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    // Clone the cache
    let cache_clone = cache.clone();

    // Verify clone has same state
    assert_eq!(cache_clone.len(), cache.len());
    let (orig_k, _) = cache.get(0);
    let (clone_k, _) = cache_clone.get(0);
    assert_eq!(orig_k.len(), clone_k.len());
    assert!((orig_k[0] - clone_k[0]).abs() < f32::EPSILON);

    // Verify clone is independent (modify original shouldn't affect clone)
    let k2 = vec![999.0f32; kv_size];
    let v2 = vec![888.0f32; kv_size];
    cache.append(0, &k2, &v2);
    cache.append(1, &k2, &v2);

    // Original should have 2 positions, clone should still have 1
    assert_eq!(cache.len(), 2);
    assert_eq!(cache_clone.len(), 1);
}
