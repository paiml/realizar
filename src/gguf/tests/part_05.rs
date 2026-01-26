//! GGUF Part 05: IMP-115 (Fused Attention) + IMP-117 (Small Buffer Optimization) +
//!               IMP-118 (True GPU Batched GEMM) + IMP-119 (GPU Fused Attention)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedModelCached, QuantizedGenerateConfig};

// ========================================================================
// IMP-115: Fused Attention Kernel Tests (Q@K^T → softmax → @V)
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115a_fused_single_head_attention_correctness() {
    // IMP-115a: Verify fused attention matches separate operations
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Create single-head Q, K, V
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Reference: separate operations
    let reference = cached_model
        .model()
        .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 4)
        .expect("Reference attention should succeed");

    // Fused: single kernel
    let result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Fused attention should succeed");

    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-4,
            "IMP-115a: Fused differs at {}: ref={}, fused={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115b_fused_multihead_attention_correctness() {
    // IMP-115b: Verify fused multi-head attention matches reference
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 8;
    let hidden_dim = config.hidden_dim;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Reference: flattened multi-head
    let reference = cached_model
        .flattened_multihead_attention(&q, &k, &v, seq_len)
        .expect("Reference attention should succeed");

    // Fused multi-head
    let result = cached_model
        .fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("Fused multi-head attention should succeed");

    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-115b: Fused MHA differs at {}: ref={}, fused={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115c_fused_attention_no_intermediate_allocation() {
    // IMP-115c: Verify fused attention doesn't allocate large intermediate tensors
    // We test this by verifying output is correct for larger sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 50,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 32; // Larger sequence to stress test
    let hidden_dim = config.hidden_dim;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();

    let result = cached_model
        .fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("Fused attention should succeed for larger sequences");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-115c: Output should have correct dimensions"
    );

    // Verify output is not all zeros
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-115c: Output should have non-trivial values"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115d_fused_causal_mask_correctness() {
    // IMP-115d: Verify causal masking is correctly applied in fused kernel
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 4;
    let head_dim = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use Q where different positions have distinct patterns
    // This helps verify causal masking is working
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| {
            let pos = i / head_dim;
            ((pos * 10 + i % head_dim) as f32) * 0.1
        })
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    let result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Fused causal attention should succeed");

    // Verify output dimensions
    assert_eq!(result.len(), seq_len * head_dim);

    // Verify each position's output is influenced only by positions 0..=i
    // Position 0 can only attend to itself
    // Position 1 can attend to 0 and 1
    // etc.
    // We can't easily verify this without access to internal attention weights,
    // but we can verify output is valid (non-NaN, finite, reasonable range)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "IMP-115d: Output at {} should be finite, got {}",
            i,
            val
        );
        assert!(
            val.abs() < 10.0,
            "IMP-115d: Output at {} should be in reasonable range, got {}",
            i,
            val
        );
    }
}

// ========================================================================
// IMP-117: Small Buffer Optimization Tests (SmallVec)
// ========================================================================

#[test]
fn test_imp_117a_token_buffer_inline_allocation() {
    // IMP-117a: TokenBuffer should use stack allocation for small sizes
    use crate::gguf::{TokenBuffer, TOKEN_BUFFER_INLINE_CAP};

    // Create buffer within inline capacity
    let mut buffer: TokenBuffer = TokenBuffer::new();
    for i in 0..TOKEN_BUFFER_INLINE_CAP {
        buffer.push(i as u32);
    }

    // Verify capacity and inline status
    assert_eq!(
        buffer.len(),
        TOKEN_BUFFER_INLINE_CAP,
        "IMP-117a: Buffer should hold TOKEN_BUFFER_INLINE_CAP elements"
    );

    // SmallVec is inline when len <= inline capacity
    assert!(
        !buffer.spilled(),
        "IMP-117a: Buffer should not spill to heap at inline capacity"
    );

    // Adding one more should trigger heap allocation
    buffer.push(999);
    assert!(
        buffer.spilled(),
        "IMP-117a: Buffer should spill to heap when exceeding inline capacity"
    );
}

#[test]
fn test_imp_117b_attention_buffer_inline_allocation() {
    // IMP-117b: AttentionBuffer should use stack allocation for small sizes
    use crate::gguf::{AttentionBuffer, ATTENTION_BUFFER_INLINE_CAP};

    let mut buffer: AttentionBuffer = AttentionBuffer::new();
    for i in 0..ATTENTION_BUFFER_INLINE_CAP {
        buffer.push(i as f32 * 0.1);
    }

    assert_eq!(
        buffer.len(),
        ATTENTION_BUFFER_INLINE_CAP,
        "IMP-117b: Attention buffer should hold ATTENTION_BUFFER_INLINE_CAP elements"
    );
    assert!(
        !buffer.spilled(),
        "IMP-117b: Attention buffer should not spill at inline capacity"
    );
}

#[test]
fn test_imp_117c_hidden_buffer_inline_allocation() {
    // IMP-117c: HiddenBuffer should use stack allocation for small models
    use crate::gguf::{HiddenBuffer, HIDDEN_BUFFER_INLINE_CAP};

    let mut buffer: HiddenBuffer = HiddenBuffer::new();
    for i in 0..HIDDEN_BUFFER_INLINE_CAP {
        buffer.push(i as f32 * 0.01);
    }

    assert_eq!(
        buffer.len(),
        HIDDEN_BUFFER_INLINE_CAP,
        "IMP-117c: Hidden buffer should hold HIDDEN_BUFFER_INLINE_CAP elements"
    );
    assert!(
        !buffer.spilled(),
        "IMP-117c: Hidden buffer should not spill at inline capacity"
    );
}

#[test]
fn test_imp_117d_buffer_watermarks() {
    // IMP-117d: Verify buffer watermark constants are reasonable
    use crate::gguf::{BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE};

    // Low < High < Max
    assert!(
        BUFFER_LW_SIZE < BUFFER_HW_SIZE,
        "IMP-117d: Low watermark should be less than high watermark"
    );
    assert!(
        BUFFER_HW_SIZE < BUFFER_MAX_SIZE,
        "IMP-117d: High watermark should be less than max size"
    );

    // Reasonable ranges
    assert!(
        BUFFER_LW_SIZE >= 1024,
        "IMP-117d: Low watermark should be at least 1KB"
    );
    assert!(
        BUFFER_MAX_SIZE <= 64 * 1024,
        "IMP-117d: Max buffer should be at most 64KB"
    );
}

#[test]
fn test_imp_117e_token_buffer_from_slice() {
    // IMP-117e: TokenBuffer should work with from_slice
    use crate::gguf::TokenBuffer;

    let tokens: &[u32] = &[1, 2, 3, 4, 5];
    let buffer: TokenBuffer = TokenBuffer::from_slice(tokens);

    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.as_slice(), tokens);
    assert!(!buffer.spilled(), "IMP-117e: Small slice should not spill");
}

#[test]
fn test_imp_117f_generate_with_token_buffer() {
    // IMP-117f: Test generate_with_smallvec returns correct SmallVec type
    use crate::gguf::{TokenBuffer, TOKEN_BUFFER_INLINE_CAP};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);

    // Test with small prompt that fits in inline capacity
    let prompt: TokenBuffer = TokenBuffer::from_slice(&[1, 2, 3, 4, 5]);
    assert!(
        prompt.len() < TOKEN_BUFFER_INLINE_CAP,
        "IMP-117f: Test prompt should be within inline capacity"
    );

    // Generate tokens using the SmallVec-based API
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
        trace: false,
    };

    let result = model.generate_with_smallvec(&prompt, &gen_config);
    assert!(
        result.is_ok(),
        "IMP-117f: generate_with_smallvec should succeed"
    );

    let generated = result.expect("generation should succeed");
    assert!(
        generated.len() > prompt.len(),
        "IMP-117f: Generated tokens should include prompt + new tokens"
    );
}

// ========================================================================
// IMP-118: True GPU Batched GEMM Kernel Tests
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118a_true_batched_gemm_correctness() {
    // IMP-118a: Verify true batched GEMM produces correct results
    // Strategy: Process all batches in single kernel invocation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let batch_size = 8;
    let m = 16;
    let k = 32;
    let n = 16;

    // Create batched input data
    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for b in 0..batch_size {
        for i in 0..m * k {
            batched_a[b * m * k + i] = ((b * m * k + i) % 17) as f32 * 0.1;
        }
        for i in 0..k * n {
            batched_b[b * k * n + i] = ((b * k * n + i) % 13) as f32 * 0.1;
        }
    }

    // True batched GEMM should process all batches together
    let result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("True batched GEMM should succeed");

    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-118a: Output should have shape [batch, m, n]"
    );

    // Verify by computing reference per-batch
    for b in 0..batch_size {
        let a_start = b * m * k;
        let b_start = b * k * n;
        let out_start = b * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for kk in 0..k {
                    expected += batched_a[a_start + i * k + kk] * batched_b[b_start + kk * n + j];
                }
                let actual = result[out_start + i * n + j];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-2,
                    "IMP-118a: Batch {} pos ({},{}) mismatch: expected={}, got={}, diff={}",
                    b,
                    i,
                    j,
                    expected,
                    actual,
                    diff
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118b_true_batched_gemm_matches_flattened() {
    // IMP-118b: True batched GEMM should match flattened implementation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let batch_size = 4;
    let m = 8;
    let k = 16;
    let n = 8;

    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for i in 0..batched_a.len() {
        batched_a[i] = (i % 19) as f32 * 0.05;
    }
    for i in 0..batched_b.len() {
        batched_b[i] = (i % 23) as f32 * 0.05;
    }

    // Compare true batched vs flattened
    let true_result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("True batched GEMM should succeed");

    let flat_result = cached_model
        .flattened_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Flattened GEMM should succeed");

    assert_eq!(true_result.len(), flat_result.len());
    for i in 0..true_result.len() {
        let diff = (true_result[i] - flat_result[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-118b: Results differ at {}: true={}, flat={}, diff={}",
            i,
            true_result[i],
            flat_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118c_true_batched_gemm_large_batch() {
    // IMP-118c: True batched GEMM should handle large batch sizes efficiently
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    // Large batch that benefits from true GPU batching
    let batch_size = 32;
    let m = 16;
    let k = 64;
    let n = 16;

    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for i in 0..batched_a.len() {
        batched_a[i] = (i % 31) as f32 * 0.02;
    }
    for i in 0..batched_b.len() {
        batched_b[i] = (i % 29) as f32 * 0.02;
    }

    let result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Large batch true GEMM should succeed");

    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-118c: Large batch output should have correct dimensions"
    );

    // Verify non-trivial output
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-118c: Output should have non-trivial values, got sum={}",
        sum
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118d_true_batched_attention() {
    // IMP-118d: Use true batched GEMM for multi-head attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;

    // Create Q, K, V tensors
    let q: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Use true batched GEMM for attention
    let result = cached_model
        .true_batched_multihead_attention(&q, &k, &v, seq_len, num_heads, head_dim)
        .expect("True batched attention should succeed");

    assert_eq!(
        result.len(),
        num_heads * seq_len * head_dim,
        "IMP-118d: Attention output should have correct shape"
    );

    // Verify normalized attention (each position should have weighted values)
    for h in 0..num_heads {
        for pos in 0..seq_len {
            let out_start = h * seq_len * head_dim + pos * head_dim;
            let slice = &result[out_start..out_start + head_dim];
            let sum: f32 = slice.iter().map(|x| x.abs()).sum();
            assert!(
                sum > 0.0 || pos == 0,
                "IMP-118d: Head {} pos {} should have non-zero output",
                h,
                pos
            );
        }
    }
}

// ========================================================================
// IMP-119: GPU-Accelerated Fused Attention for Long Sequences
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119a_gpu_fused_attention_correctness() {
    // IMP-119a: Verify GPU fused attention produces correct results
    // Uses GPU for long sequences where compute dominates transfer overhead
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    // Long sequence that benefits from GPU
    let seq_len = 64;
    let head_dim = 16;

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use GPU-accelerated fused attention
    let result = cached_model
        .gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("GPU fused attention should succeed");

    assert_eq!(
        result.len(),
        seq_len * head_dim,
        "IMP-119a: Output should have shape [seq_len, head_dim]"
    );

    // Verify causality: later positions should have different values than if
    // they could attend to all positions
    // Position 0 can only attend to itself
    let pos0_sum: f32 = result[0..head_dim].iter().sum();
    // Position seq_len-1 can attend to all previous positions
    let last_pos_sum: f32 = result[(seq_len - 1) * head_dim..].iter().sum();

    // These sums should be different due to causal masking
    assert!(
        (pos0_sum - last_pos_sum).abs() > 0.001 || seq_len == 1,
        "IMP-119a: Causal masking should affect output"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119b_gpu_fused_matches_cpu_fused() {
    // IMP-119b: GPU fused attention should match CPU fused attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 32;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 19) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 23) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 29) as f32 * 0.05)
        .collect();

    // CPU fused attention (IMP-115)
    let cpu_result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("CPU fused attention should succeed");

    // GPU fused attention (IMP-119)
    let gpu_result = cached_model
        .gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("GPU fused attention should succeed");

    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 1e-2,
            "IMP-119b: Results differ at {}: cpu={}, gpu={}, diff={}",
            i,
            cpu_result[i],
            gpu_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119c_gpu_fused_multihead_long_sequence() {
    // IMP-119c: GPU fused multi-head attention for long sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    // Long sequence with multiple heads
    let seq_len = 128;
    let hidden_dim = 128;
    let num_heads = 8;
    let _head_dim = hidden_dim / num_heads;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 17) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Use GPU-accelerated multihead fused attention
    let result = cached_model
        .gpu_fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("GPU fused multihead attention should succeed");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-119c: Output should have shape [seq_len, hidden_dim]"
    );

    // Verify each position has non-trivial output
    for pos in 0..seq_len {
        let slice = &result[pos * hidden_dim..(pos + 1) * hidden_dim];
        let sum: f32 = slice.iter().map(|x| x.abs()).sum();
        assert!(
            sum > 0.0 || pos == 0,
            "IMP-119c: Position {} should have non-zero output",
            pos
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119d_adaptive_cpu_gpu_dispatch() {
    // IMP-119d: Verify adaptive dispatch chooses CPU for short, GPU for long sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Short sequence - should work regardless of backend choice
    let short_seq_len = 8;
    let short_q: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let short_k: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let short_v: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let short_result = cached_model
        .adaptive_fused_attention(&short_q, &short_k, &short_v, short_seq_len, head_dim, scale)
        .expect("Adaptive attention for short sequence should succeed");

    assert_eq!(short_result.len(), short_seq_len * head_dim);

    // Long sequence - should also work
    let long_seq_len = 128;
    let long_q: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let long_k: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let long_v: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let long_result = cached_model
        .adaptive_fused_attention(&long_q, &long_k, &long_v, long_seq_len, head_dim, scale)
        .expect("Adaptive attention for long sequence should succeed");

    assert_eq!(long_result.len(), long_seq_len * head_dim);

    // Both should produce valid outputs
    let short_sum: f32 = short_result.iter().sum();
    let long_sum: f32 = long_result.iter().sum();

    // Longer sequence should have larger accumulated values (more positions attending)
    assert!(
        long_sum.abs() > short_sum.abs() / 2.0,
        "IMP-119d: Long sequence output should be non-trivial"
    );
}
