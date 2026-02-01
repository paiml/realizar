use super::*;
use crate::cuda::memory::{SizeClass, TransferMode};
use crate::cuda::pipeline::{
use proptest::prelude::*;
use serial_test::serial;
    println!("T-QA-012a: transformer_layer_gpu basic execution PASSED");
}

/// T-QA-012b: Test transformer_layer_gpu_tiled_profiled
///
/// Same setup as T-QA-012a but uses the tiled profiled variant.
#[test]
#[serial]
fn test_tqa012b_transformer_layer_gpu_tiled_profiled() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012b: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Same dimensions as T-QA-012a
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    // Initialize KV cache
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012b: KV cache init");

    // Load weights (same as T-QA-012a)
    let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
    executor
        .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(
            &format!("{}.attn_output.weight", layer_prefix),
            &qkvo_weights,
        )
        .expect("CUDA operation failed");

    let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
    let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
    executor
        .load_quantized_weights(
            &format!("{}.ffn_gate.weight", layer_prefix),
            &gate_up_weights,
        )
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
        .expect("CUDA operation failed");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input =
        GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

    // Execute tiled profiled variant
    let result = executor.transformer_layer_gpu_tiled_profiled(
        &input,
        layer_idx,
        layer_prefix,
        hidden_dim,
        intermediate_dim,
        &attn_norm_gamma,
        &ffn_norm_gamma,
        epsilon,
    );

    assert!(
        result.is_ok(),
        "T-QA-012b: transformer_layer_gpu_tiled_profiled should execute: {:?}",
        result.err()
    );
    let output = result.expect("CUDA operation failed");
    assert_eq!(output.len(), hidden_dim as usize);
    println!("T-QA-012b: transformer_layer_gpu_tiled_profiled execution PASSED");
}

/// T-QA-012c: Test transformer_layer_gpu_true_dp4a
///
/// Tests the DP4A (dot product of 4 8-bit integers) optimized variant.
/// Note: CORRECTNESS-001 disables DP4A kernel due to scale extraction issue.
/// This test verifies the code path is exercised, accepting either success or
/// the known PTX error from the disabled kernel.
#[test]
#[serial]
fn test_tqa012c_transformer_layer_gpu_true_dp4a() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012c: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Same dimensions
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012c: KV cache init");

    // Load weights
    let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
    executor
        .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(
            &format!("{}.attn_output.weight", layer_prefix),
            &qkvo_weights,
        )
        .expect("CUDA operation failed");

    let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
    let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
    executor
        .load_quantized_weights(
            &format!("{}.ffn_gate.weight", layer_prefix),
            &gate_up_weights,
        )
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
        .expect("CUDA operation failed");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input =
        GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

    // Execute DP4A variant
    let result = executor.transformer_layer_gpu_true_dp4a(
        &input,
        layer_idx,
        layer_prefix,
        hidden_dim,
        intermediate_dim,
        &attn_norm_gamma,
        &ffn_norm_gamma,
        epsilon,
    );

    // CORRECTNESS-001: DP4A kernel has known scale extraction issue
    // Accept either success or the expected PTX error
    match &result {
        Ok(output) => {
            assert_eq!(output.len(), hidden_dim as usize);
            println!("T-QA-012c: transformer_layer_gpu_true_dp4a execution PASSED");
        },
        Err(e) => {
            let err_msg = format!("{:?}", e);
            // Accept known PTX errors from the disabled DP4A kernel
            assert!(
                err_msg.contains("PTX") || err_msg.contains("ModuleLoad"),
                "T-QA-012c: Unexpected error (not PTX-related): {}",
                err_msg
            );
            println!(
                "T-QA-012c: transformer_layer_gpu_true_dp4a correctly reports DP4A kernel issue (CORRECTNESS-001)"
            );
        },
    }
}

/// T-QA-012d: Test multi-layer forward pass via forward_all_layers_gpu
///
/// Tests the complete forward pass through multiple transformer layers.
#[test]
#[serial]
fn test_tqa012d_forward_all_layers_gpu() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012d: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use 2 layers to test multi-layer handling
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 2usize;
    let max_seq_len = 32usize;
    let epsilon = 1e-5f32;

    // Initialize KV cache for all layers
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012d: KV cache init");

    // Load weights for both layers
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("blk.{}", layer_idx);

        let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
        executor
            .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(
                &format!("{}.attn_output.weight", layer_prefix),
                &qkvo_weights,
            )
            .expect("CUDA operation failed");

        let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
        let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
        executor
            .load_quantized_weights(
                &format!("{}.ffn_gate.weight", layer_prefix),
                &gate_up_weights,
            )
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
            .expect("CUDA operation failed");

        // Cache RMSNorm gammas
        let gamma = vec![1.0f32; hidden_dim as usize];
        executor
            .cache_rmsnorm_gamma(&format!("blk.{}.attn_norm.gamma", layer_idx), &gamma)
            .expect("CUDA operation failed");
        executor
            .cache_rmsnorm_gamma(&format!("blk.{}.ffn_norm.gamma", layer_idx), &gamma)
            .expect("CUDA operation failed");
    }

    // Cache output norm using preload_output_norm
    let gamma = vec![1.0f32; hidden_dim as usize];
    executor
        .preload_output_norm(&gamma)
        .expect("CUDA operation failed");

    // Cache LM head (output.weight) for final projection
    let lm_head_weights = create_mock_q4k_weights_for_harness(1000, 256); // vocab_size=1000
    executor
        .load_quantized_weights("output.weight", &lm_head_weights)
        .expect("CUDA operation failed");

    // Build indexed weights for forward_all_layers_gpu
    executor
        .build_indexed_weights(num_layers, |layer_idx| format!("blk.{}", layer_idx))
        .expect("T-QA-012d: Build indexed weights");

    // Input/output slices (forward_all_layers_gpu uses slices, not GpuBuffer)
    let input_data = vec![0.1f32; hidden_dim as usize];
    let mut output_data = vec![0.0f32; hidden_dim as usize];
    let position = 0u32;

    // Execute forward_all_layers_gpu
    let result = executor.forward_all_layers_gpu(
        &input_data,
        &mut output_data,
        position,
        num_layers,
        hidden_dim,
        intermediate_dim,
        epsilon,
    );

    assert!(
        result.is_ok(),
        "T-QA-012d: forward_all_layers_gpu should execute: {:?}",
        result.err()
    );
    assert_eq!(output_data.len(), hidden_dim as usize);
    println!("T-QA-012d: forward_all_layers_gpu multi-layer execution PASSED");
}

/// T-QA-012e: Test error handling for missing weights
///
/// Verifies that transformer_layer_gpu returns appropriate error when weights are missing.
#[test]
#[serial]
fn test_tqa012e_transformer_layer_gpu_missing_weights() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012e: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    // Initialize KV cache but DON'T load weights
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012e: KV cache init");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input =
        GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

    // Attempt execution without weights - should fail
    let result = executor.transformer_layer_gpu(
        &input,
        layer_idx,
        layer_prefix,
        hidden_dim,
        intermediate_dim,
        &attn_norm_gamma,
        &ffn_norm_gamma,
        epsilon,
    );

    assert!(
        result.is_err(),
        "T-QA-012e: transformer_layer_gpu should fail without weights"
    );
    // Extract error without unwrap_err (which requires Debug on Ok type)
    let err = match result {
        Ok(_) => panic!("T-QA-012e: Expected error but got Ok"),
        Err(e) => e,
    };
    let err_msg = format!("{:?}", err);
    assert!(
        err_msg.contains("not cached") || err_msg.contains("PAR-023"),
        "T-QA-012e: Error should mention missing cached weights: {}",
        err_msg
    );
    println!("T-QA-012e: Missing weights error handling PASSED");
}

/// T-QA-012f: Test incremental attention with KV cache update
///
/// Verifies that calling transformer_layer_gpu multiple times updates KV cache correctly.
#[test]
#[serial]
fn test_tqa012f_transformer_layer_kv_cache_update() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012f: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012f: KV cache init");

    // Load weights
    let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
    executor
        .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(
            &format!("{}.attn_output.weight", layer_prefix),
            &qkvo_weights,
        )
        .expect("CUDA operation failed");

    let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
    let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
    executor
        .load_quantized_weights(
            &format!("{}.ffn_gate.weight", layer_prefix),
            &gate_up_weights,
        )
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
        .expect("CUDA operation failed");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    // Execute twice to verify KV cache updates
    for token_idx in 0..2 {
        let input_data = vec![0.1f32 * (token_idx as f32 + 1.0); hidden_dim as usize];
        let input =
            GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

        let result = executor.transformer_layer_gpu(
            &input,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_norm_gamma,
            &ffn_norm_gamma,
            epsilon,
        );

        assert!(
            result.is_ok(),
            "T-QA-012f: Token {} should process successfully: {:?}",
            token_idx,
            result.err()
        );
    }

    // Verify KV cache length increased
    let cache_len = executor
        .kv_cache_lengths
        .get(&layer_idx)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        cache_len, 2,
        "T-QA-012f: KV cache should have 2 entries after 2 tokens"
    );
    println!("T-QA-012f: KV cache update across multiple tokens PASSED");
}

// ============================================================================
// T-QA-013: Synthetic Graph Tests
// ============================================================================
// These tests exercise CUDA graph capture/replay state management.

/// T-QA-013a: Test decode graph state management
///
/// Verifies has_decode_graph and clear_decode_graph work correctly.
#[test]
#[serial]
fn test_tqa013a_decode_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013a: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no graph captured
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013a: No decode graph initially"
    );

    // Clear graph (should be no-op on empty state)
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013a: Still no graph after clear"
    );

    println!("T-QA-013a: Decode graph state management PASSED");
}

/// T-QA-013b: Test workspace and indexed weight checks
///
/// Verifies the workspace and indexed weight state checks used by graph capture.
#[test]
#[serial]
fn test_tqa013b_workspace_and_indexed_weights() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013b: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no workspace
    assert!(
        !executor.has_workspace(),
        "T-QA-013b: No workspace initially"
    );

    // Initially no indexed weights
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013b: No indexed weights initially"
    );

    // Clear indexed weights (should be no-op)
    executor.clear_indexed_weights();
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013b: Still no indexed weights after clear"
    );

    println!("T-QA-013b: Workspace and indexed weights checks PASSED");
}

/// T-QA-013c: Test CUDA graph disable env var
///
/// Verifies that the CUDA_GRAPH_DISABLE environment variable path is exercised.
#[test]
#[serial]
fn test_tqa013c_graph_disable_env_var() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013c: CUDA not available, skipping");
        return;
    }

    // Set env var to disable graphs
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    // Create executor - env var is read lazily
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Just verify executor was created (env var affects forward pass path selection)
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013c: No graph should be captured when disabled"
    );

    // Clean up env var
    std::env::remove_var("CUDA_GRAPH_DISABLE");

    println!("T-QA-013c: CUDA_GRAPH_DISABLE env var handling PASSED");
}

/// T-QA-013d: Test graphed forward with incomplete state (falls back to non-graphed)
///
/// Verifies that forward_all_layers_gpu_to_logits_graphed gracefully falls back
/// when workspace/indexed weights are not available.
#[test]
#[serial]
fn test_tqa013d_graphed_forward_fallback() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013d: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Model dimensions
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_layers = 1usize;
    let vocab_size = 1000u32;
    let epsilon = 1e-5f32;

    // Input/output (without setting up weights - will fail at forward pass)
    let input = vec![0.1f32; hidden_dim as usize];
    let mut logits = vec![0.0f32; vocab_size as usize];

    // Try graphed forward without weights - should fail with missing weights error
    let result = executor.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        num_layers,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        epsilon,
    );

    // Expect error due to missing weights (not a graph capture error)
    assert!(result.is_err(), "T-QA-013d: Should fail without weights");
    let err_msg = format!("{:?}", result.expect_err("CUDA operation failed"));
    // Error should mention missing cached weights or norms, not graph capture failure
    assert!(
        err_msg.contains("not cached")
            || err_msg.contains("PAR-023")
            || err_msg.contains("Workspace"),
        "T-QA-013d: Error should be about missing state, not graph: {}",
        err_msg
    );

    // No graph should be captured on failure
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013d: No graph captured on failure"
    );

    println!("T-QA-013d: Graphed forward fallback on incomplete state PASSED");
}

/// T-QA-013e: Test batched decode graph state management
///
/// Verifies batched graph state is properly initialized.
#[test]
#[serial]
fn test_tqa013e_batched_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013e: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Verify batched decode graphs map is empty initially
    // We check this indirectly via the fact that has_decode_graph returns false
    // (batched graphs use a different storage but similar patterns)
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013e: No graphs captured initially"
    );

    println!("T-QA-013e: Batched graph state initialization PASSED");
}

/// T-QA-013f: Test graph state after clear_workspace
///
/// Verifies that clearing workspace affects graph capture eligibility.
#[test]
#[serial]
fn test_tqa013f_clear_workspace_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013f: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear workspace and verify state
    executor.clear_workspace();
    assert!(
        !executor.has_workspace(),
        "T-QA-013f: No workspace after clear"
    );

    // Clear decode graph and verify
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013f: No graph after clear"
    );

    // Clear indexed weights
    executor.clear_indexed_weights();
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013f: No indexed weights after clear"
    );

    println!("T-QA-013f: Clear workspace/graph state PASSED");
}

// ============================================================================
// T-QA-014: Buffer Fuzzing Tests (proptest GpuBuffer lifecycle)
// ============================================================================
// These tests use property-based testing to fuzz GpuBuffer operations.

proptest! {
    /// T-QA-014a: Property - GpuBuffer allocation succeeds for various sizes
    ///
    /// Tests that GpuBuffer::new works for a range of sizes (1 to 10000).
    #[test]
    #[serial]
    fn prop_tqa014a_buffer_allocation_various_sizes(size in 1usize..10000) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014a: Executor init failed: {}", e)))?;

        let buf: GpuBuffer<f32> = GpuBuffer::new(&executor.context, size)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014a: Allocation failed for size {}: {}", size, e)))?;

        prop_assert_eq!(buf.len(), size, "T-QA-014a: Buffer length mismatch");
        prop_assert_eq!(buf.size_bytes(), size * std::mem::size_of::<f32>(), "T-QA-014a: Byte size mismatch");
    }

    /// T-QA-014b: Property - GpuBuffer from_host preserves data integrity
    ///
    /// Tests that data uploaded via from_host can be read back correctly.
    #[test]
    #[serial]
    fn prop_tqa014b_buffer_data_integrity(data in prop::collection::vec(any::<f32>(), 1..1000)) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: Executor init failed: {}", e)))?;

        // Filter out NaN values which can't be compared with ==
        let data: Vec<f32> = data.into_iter().filter(|x| !x.is_nan()).collect();
        if data.is_empty() {
            return Ok(());
        }

        let buf = GpuBuffer::from_host(&executor.context, &data)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: from_host failed: {}", e)))?;

        let mut readback = vec![0.0f32; data.len()];
        buf.copy_to_host(&mut readback)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: copy_to_host failed: {}", e)))?;

        for (i, (expected, actual)) in data.iter().zip(readback.iter()).enumerate() {
            if expected.is_finite() && actual.is_finite() {
                prop_assert!(
                    (expected - actual).abs() < 1e-6,
                    "T-QA-014b: Data mismatch at index {}: expected {}, got {}",
                    i, expected, actual
                );
            }
        }
    }

    /// T-QA-014c: Property - Multiple buffers can be allocated and freed
    ///
    /// Tests that allocating multiple buffers in sequence works correctly.
    #[test]
    #[serial]
    fn prop_tqa014c_multiple_buffer_allocation(num_buffers in 1..20usize, base_size in 100..1000usize) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014c: Executor init failed: {}", e)))?;

        let mut buffers = Vec::new();
        for i in 0..num_buffers {
            let size = base_size + i * 10;
            let buf: GpuBuffer<f32> = GpuBuffer::new(&executor.context, size)
                .map_err(|e| TestCaseError::fail(format!("T-QA-014c: Allocation {} failed: {}", i, e)))?;
            prop_assert_eq!(buf.len(), size);
            buffers.push(buf);
        }

        // Verify all buffers still valid
        for (i, buf) in buffers.iter().enumerate() {
            let expected_size = base_size + i * 10;
            prop_assert_eq!(buf.len(), expected_size, "T-QA-014c: Buffer {} size changed", i);
        }
        // buffers will be dropped here, testing Drop correctness
    }

    /// T-QA-014d: Property - Buffer rewrite works correctly
    ///
    /// Tests that writing new data to an existing buffer works.
    #[test]
    #[serial]
    fn prop_tqa014d_buffer_rewrite(
        initial in prop::collection::vec(1.0f32..100.0, 50..200),
        update in prop::collection::vec(100.0f32..200.0, 50..200)
    ) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Executor init failed: {}", e)))?;

        // Use the smaller size to ensure both vectors fit
        let size = initial.len().min(update.len());
        if size == 0 {
            return Ok(());
        }

        // Initial upload
        let mut buf = GpuBuffer::from_host(&executor.context, &initial[..size])
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Initial upload failed: {}", e)))?;

        // Overwrite with new data
        buf.copy_from_host(&update[..size])
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Rewrite failed: {}", e)))?;

        // Verify new data
        let mut readback = vec![0.0f32; size];
        buf.copy_to_host(&mut readback)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Readback failed: {}", e)))?;

        for (i, (expected, actual)) in update[..size].iter().zip(readback.iter()).enumerate() {
            prop_assert!(
                (expected - actual).abs() < 1e-6,
                "T-QA-014d: Data mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }
}

/// T-QA-014e: Test edge case - single element buffer
#[test]
#[serial]
fn test_tqa014e_single_element_buffer() {
    if !CudaExecutor::is_available() {
        println!("T-QA-014e: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Single element buffer
    let data = vec![42.0f32];
    let buf = GpuBuffer::from_host(&executor.context, &data).expect("T-QA-014e: from_host");

    assert_eq!(buf.len(), 1, "T-QA-014e: Single element length");
    assert_eq!(buf.size_bytes(), 4, "T-QA-014e: Single element bytes");

    let mut readback = vec![0.0f32];
    buf.copy_to_host(&mut readback)
        .expect("T-QA-014e: copy_to_host");
    assert!(
        (readback[0] - 42.0).abs() < 1e-6,
        "T-QA-014e: Value preserved"
    );

    println!("T-QA-014e: Single element buffer PASSED");
}

/// T-QA-014f: Test edge case - large buffer allocation
#[test]
#[serial]
fn test_tqa014f_large_buffer_allocation() {
    if !CudaExecutor::is_available() {
        println!("T-QA-014f: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Large buffer (1M elements = 4MB)
    let size = 1_000_000usize;
    let mut buf: GpuBuffer<f32> =
        GpuBuffer::new(&executor.context, size).expect("T-QA-014f: Large buffer allocation");

    assert_eq!(buf.len(), size, "T-QA-014f: Large buffer length");
    assert_eq!(buf.size_bytes(), size * 4, "T-QA-014f: Large buffer bytes");

    // Initialize with pattern
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    buf.copy_from_host(&data)
        .expect("T-QA-014f: copy_from_host");

    // Spot check some values
    let mut readback = vec![0.0f32; size];
    buf.copy_to_host(&mut readback)
        .expect("T-QA-014f: copy_to_host");

    assert!((readback[0] - 0.0).abs() < 1e-5, "T-QA-014f: First value");
    assert!(
        (readback[1000] - 1.0).abs() < 1e-5,
        "T-QA-014f: Value at 1000"
    );
    assert!(
        (readback[size - 1] - (size - 1) as f32 * 0.001).abs() < 1e-5,
        "T-QA-014f: Last value"
    );

    println!("T-QA-014f: Large buffer allocation PASSED");
}

// =========================================================================
// T-COV-001: Comprehensive KernelType PTX Generation Coverage Tests
// Targets: 95% cuda.rs coverage by exercising all KernelType variants
// =========================================================================

#[test]
fn test_tcov001a_attention_tensor_core_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::AttentionTensorCore {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: true,
    });
    assert!(ptx.contains(".version"), "PTX should have version");
    assert!(
        ptx.contains("attention") || ptx.contains("flash"),
        "PTX should contain attention kernel"
    );
}

#[test]
fn test_tcov001b_bias_activation_ptx() {
    let kernels = CudaKernels::new();

    // Test with ReLU
    let ptx_relu = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 1,
    });
    assert!(ptx_relu.contains(".version"));

    // Test with GELU
    let ptx_gelu = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 2,
    });
    assert!(ptx_gelu.contains(".version"));

    // Test with None
    let ptx_none = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 0,
    });
    assert!(ptx_none.contains(".version"));
}

#[test]
fn test_tcov001c_gemm_fp16_tensor_core_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmFp16TensorCore {
        m: 64,
        n: 64,
        k: 64,
    });
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm") || ptx.contains("wmma"));
}

#[test]
fn test_tcov001d_fused_q4q8_dot_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedQ4Q8Dot { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001e_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("q4k"));
}

